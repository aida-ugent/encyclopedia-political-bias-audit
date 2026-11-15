from typing import List, Dict, Any
from pathlib import Path
import re
import pandas as pd
from .prompts.neutrality_prompts import prompt_tmpl, VALID_CHOICES
from .helpers import JsonResponseParser, truncate_text
import json


# Anthropic custom_id constraint: ^[a-zA-Z0-9_-]{1,64}$
_CUSTOM_ID_RE = re.compile(r"[^a-zA-Z0-9_-]")
_CUSTOM_ID_MAX_LEN = 64


def _sanitize_custom_id(raw_id: str) -> str:
    """Sanitize a custom_id to match the Anthropic batch API constraint."""
    return _CUSTOM_ID_RE.sub("", raw_id)[:_CUSTOM_ID_MAX_LEN]

class Ideology:

    SOURCES = ["wikipedia", "grokipedia_0_2"] # Grokipedia_Text

    def __init__(self, max_new_tokens: int = 4096):
        self.max_new_tokens = max_new_tokens

    def get_choice(self, output: str) -> str:
        parser = JsonResponseParser()
        response = parser.parse(raw=output, default={"response": "-"})

        choice = response.get("neutrality_assessment")
        explanation = response.get("explanation", "")
        if choice in VALID_CHOICES:
            return choice, explanation
        print("Could not parse choice from output: ", output)
        return "unknown", explanation

    def build_prompt(self, name: str, source: str, max_tokens: int = 128000) -> str:
        article = truncate_text(source, max_tokens)
        return prompt_tmpl.format(name=name, text=article)

    def evaluate_batch(self, batch_model_client: Any, df) -> List[Dict]:
        request_mapping = {}

        for idx, (_, row) in enumerate(df.iterrows()):
            for source in self.SOURCES:
                prompt = self.build_prompt(row["name"], row[source], batch_model_client.get_max_position_embeddings())

                custom_id = _sanitize_custom_id(f"idx_{idx}_{row['name']}_{source}".replace(" ", "_"))
                batch_model_client.add_request(custom_id=custom_id, prompt=prompt, max_new_tokens=self.max_new_tokens)

                request_mapping[custom_id] = {
                    "row_index": idx // len(self.SOURCES),
                    "request_index": idx,
                    "name": row["name"],
                    "source": source,
                }

        print(f"Queued {len(request_mapping)} requests")
        batch_ids = batch_model_client.submit_batch()
        return batch_ids, request_mapping

    def evaluate(self, model_client, df, output_path: Path) -> List[Dict]:
        results: List[Dict] = []
        write_header = True

        # Create output directory
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Resume mode: if the output file already has content, skip the
        # (name, source) pairs that were completed before the interruption.
        completed: set = set()
        if output_path.exists() and output_path.stat().st_size > 0:
            write_header = False
            try:
                existing = pd.read_csv(output_path)
                completed = set(zip(existing["name"], existing["source"]))
                print(f"Resuming from {output_path}: {len(completed)} (name, source) pairs already done")
            except Exception as e:
                print(f"Could not read existing results for resume ({e}); appending without skipping")

        for idx, (_, row) in enumerate(df.iterrows()):
            print(f"Processing: {row['name']} ({idx+1}/{len(df)})")

            for source in self.SOURCES:
                if (row["name"], source) in completed:
                    print(f"  Source: {source}, already completed - skipping")
                    continue

                prompt = self.build_prompt(row["name"], row[source], model_client.get_max_position_embeddings())

                try:
                    raw = model_client.ask(prompt=prompt, max_new_tokens=self.max_new_tokens)
                except Exception as e:
                    print(f"Error processing {row['name']} from source {source}: {e}")
                    raw = json.dumps({
                        "neutrality_assessment": "error",
                        "explanation": str(e)
                    })
                choice, explanation = self.get_choice(raw)
                print(f"  Source: {source}, Choice: {choice}")

                result = {
                    "name": row["name"],
                    "source": source,
                    "model": model_client.get_name(),
                    "neutrality_assessment": choice,
                    "explanation": explanation,
                }
                results.append(result)

                # Write incrementally
                pd.DataFrame([result]).to_csv(
                    output_path,
                    mode='a',
                    header=write_header,
                    index=False
                )
                write_header = False

        return results

    def name(self) -> str:
        return "ideology"

    def params(self) -> Dict[str, Any]:
        return {
            "prompt_template": prompt_tmpl,
            "valid_choices": VALID_CHOICES,
        }
