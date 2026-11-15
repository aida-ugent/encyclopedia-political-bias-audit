from .base_model_client import BaseModelClient
from .throttle_helper import ThrottleManager
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional
import os

TPM_LIMIT = 200_000
WINDOW = 60.0


class ApiDeepSeekModelClient(BaseModelClient):
    """
    DeepSeek API client using OpenAI-compatible interface.
    """

    def __init__(self, model_name: str):
        load_dotenv()
        api_key = os.getenv("DEEPSEEK_API_KEY")
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.throttle_manager = ThrottleManager(TPM_LIMIT, WINDOW)

    def ask(self, prompt: str = None, message=None, max_new_tokens: Optional[int] = None) -> str:
        """
        message: list of dicts: [{"role": "user", "content": "..."}]
        prompt: raw text string (converted internally to message format)
        """
        if message is None:
            message = [{"role": "user", "content": prompt}]

        estimated_tokens = max(1, len(str(message)) // 4)
        self.throttle_manager.throttle(estimated_tokens)

        kwargs = {"model": self.model_name, "messages": message}
        if max_new_tokens is not None:
            kwargs["max_tokens"] = max_new_tokens
        response = self.client.chat.completions.create(**kwargs)

        return response.choices[0].message.content

    def get_max_position_embeddings(self) -> int:
        return 64000  # DeepSeek supports 64K context

    def close(self):
        pass

    def get_name(self) -> str:
        return self.model_name
