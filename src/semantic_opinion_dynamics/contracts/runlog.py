from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .opinion import OpinionState, OpinionUpdate


class RunMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    created_at_utc: str
    engine_version: str

    config_path: str
    network_path: str
    personas_path: str


class PromptLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system: str
    user: str
    json_schema: Dict[str, Any]


class AgentLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str
    display_name: str

    state: OpinionState

    neighbors: List[str] = Field(default_factory=list)
    llm_update: Optional[OpinionUpdate] = None
    prompt: Optional[PromptLog] = None
    error: Optional[str] = None


class StepLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: int = Field(ge=0)
    agents: Dict[str, AgentLog]
