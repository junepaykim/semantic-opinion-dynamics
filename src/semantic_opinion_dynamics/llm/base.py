from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class LLMError(RuntimeError):
    pass


class LLMClient(ABC):
    def __init__(self, *, model: str, temperature: float = 0.7, seed: Optional[int] = None) -> None:
        self.model = model
        self.temperature = temperature
        self.seed = seed

    @abstractmethod
    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError
