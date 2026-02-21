from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RunSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="run", min_length=1, max_length=200)
    output_dir: str = Field(default="runs", min_length=1, max_length=500)


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_question: str = Field(min_length=1, max_length=2000)
    steps: int = Field(default=5, ge=1, le=500)

    include_neighbor_opinion_text: bool = True
    max_neighbors_in_prompt: int = Field(default=8, ge=0, le=50)
    store_prompts: bool = False


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(default="dummy", min_length=1, max_length=50)
    model: str = Field(default="dummy-v0", min_length=1, max_length=200)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    seed: Optional[int] = None

    max_concurrency: int = Field(default=8, ge=1, le=128)
    min_interval_sec: float = Field(default=0.0, ge=0.0, le=10.0)
    max_retries: int = Field(default=2, ge=0, le=10)


class PromptsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    behavior_rules: List[str] = Field(default_factory=list, max_length=50)


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunSection
    experiment: SimulationConfig
    llm: LLMConfig
    prompts: PromptsConfig
