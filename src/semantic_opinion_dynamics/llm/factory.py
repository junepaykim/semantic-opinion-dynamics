from __future__ import annotations

from semantic_opinion_dynamics.contracts import LLMConfig
from semantic_opinion_dynamics.llm.base import LLMClient
from semantic_opinion_dynamics.llm.dummy import DummyLLMClient


def create_llm_client(cfg: LLMConfig) -> LLMClient:
    provider = cfg.provider.lower().strip()
    if provider == "dummy":
        return DummyLLMClient(model=cfg.model, temperature=cfg.temperature, seed=cfg.seed)

    if provider in {"openai", "gemini"}:
        raise NotImplementedError(
            f"LLM provider '{provider}' is a stub in this project scaffold. "
            "Switch llm.provider to 'dummy' in config.yaml, or implement a real client in semantic_opinion_dynamics/llm/."
        )

    raise ValueError(f"Unknown llm.provider: {cfg.provider}")
