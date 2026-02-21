from __future__ import annotations

from typing import Any, Dict

from semantic_opinion_dynamics.llm.base import LLMClient
from semantic_opinion_dynamics.llm.rate_limit import AsyncRateLimiter


class RateLimitedLLMClient(LLMClient):
    def __init__(self, inner: LLMClient, limiter: AsyncRateLimiter) -> None:
        super().__init__(model=inner.model, temperature=inner.temperature, seed=inner.seed)
        self._inner = inner
        self._limiter = limiter

    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        async with self._limiter:
            return await self._inner.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=json_schema,
            )
