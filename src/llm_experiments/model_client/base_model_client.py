from abc import ABC, abstractmethod
from typing import Optional


class BaseModelClient(ABC):
    @abstractmethod
    def ask(self, prompt: str = None, message=None, max_new_tokens: Optional[int] = None) -> str:
        pass

    @abstractmethod
    def get_max_position_embeddings(self) -> int:
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def get_name(self) -> str:
        return self.model_name


def create_api_model_client(model_name: str, path: str) -> BaseModelClient:
    if path == "API-DEEPSEEK":
        from .api_deepseek_client import ApiDeepSeekModelClient
        return ApiDeepSeekModelClient(model_name=model_name)
    raise ValueError(f"Unsupported study provider: {path}")
