from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):

    @abstractmethod
    def generate(
        self,
        model: str,
        system_instruction: str,
        user_prompt: str,
        response_model: Any,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def list_models(self) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        raise NotImplementedError