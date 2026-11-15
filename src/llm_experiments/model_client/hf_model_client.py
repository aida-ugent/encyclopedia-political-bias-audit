from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
)
import torch
import gc
from .base_model_client import BaseModelClient

torch.set_float32_matmul_precision("high")

class HFModelClient(BaseModelClient):
    def __init__(self, model_name: str, model_path: str, dtype=torch.bfloat16):
        self.model_name = model_name
        self.model_path = model_path
        self.tok = AutoTokenizer.from_pretrained(model_path)

        # Multimodal checkpoints (e.g. Mistral3 / Mistral-Medium-3.5) carry a
        # vision tower and register under AutoModelForImageTextToText, not the
        # causal-LM mapping. Pick the matching auto class from the config so a
        # single client supports both text-only and vision-language models.
        config = AutoConfig.from_pretrained(model_path)
        arch = (getattr(config, "architectures", None) or [""])[0]
        is_multimodal = hasattr(config, "vision_config") or "ConditionalGeneration" in arch
        model_cls = AutoModelForImageTextToText if is_multimodal else AutoModelForCausalLM

        self.model = model_cls.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto"
        )
        self.device = next(self.model.parameters()).device

        # Some checkpoints (e.g. Mistral-Medium-3.5) ship a default `max_length`
        # in generation_config. Clearing it lets `max_new_tokens` govern length
        # cleanly and silences the "both max_new_tokens and max_length set" warning.
        self.model.generation_config.max_length = None

        # Reasoning models (e.g. Mistral-Medium-3.5 with reasoning_effort="high")
        # wrap chain-of-thought in [THINK]...[/THINK] special tokens. Record the
        # closing token id so `ask()` can split reasoning from the final answer;
        # None when the model has no such marker.
        think_end_id = self.tok.convert_tokens_to_ids("[/THINK]")
        if think_end_id is None or think_end_id == self.tok.unk_token_id:
            think_end_id = None
        self.think_end_id = think_end_id

    def ask(self, prompt: str, max_new_tokens: int = 1, reasoning_effort: str = "high",
            return_reasoning: bool = False):
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort=reasoning_effort,
        ).to(self.device)

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tok.eos_token_id,
            do_sample=True,
            temperature=1,
            top_p=1,
        )

        input_len = inputs["input_ids"].shape[1]
        gen_ids = out[0][input_len:].tolist()

        # Split off the reasoning block: everything up to and including the final
        # [/THINK] is chain-of-thought; the answer is what follows. Falls back to
        # the full output when the model emits no reasoning marker.
        reasoning_ids, answer_ids = [], gen_ids
        if self.think_end_id is not None and self.think_end_id in gen_ids:
            split = len(gen_ids) - 1 - gen_ids[::-1].index(self.think_end_id)
            reasoning_ids, answer_ids = gen_ids[:split], gen_ids[split + 1:]

        answer = self.tok.decode(answer_ids, skip_special_tokens=True).strip()
        if return_reasoning:
            reasoning = self.tok.decode(reasoning_ids, skip_special_tokens=True).strip()
            return answer, reasoning
        return answer

    def get_name(self) -> str:
        return self.model_name

    def get_max_position_embeddings(self) -> int:
        # Multimodal configs (e.g. Mistral3) keep text-model fields under
        # `text_config` rather than at the top level.
        config = self.model.config
        if hasattr(config, "max_position_embeddings"):
            return config.max_position_embeddings
        return config.text_config.max_position_embeddings

    def close(self):
        try:
            del self.model
            del self.tok
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
