from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, ConfigDict, Field

from .opinion import OpinionState


class PersonaSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    display_name: str = Field(min_length=1, max_length=200)
    persona: str = Field(min_length=1, max_length=4000)
    initial_opinion: OpinionState


class PersonasSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agents: Dict[str, PersonaSpec]
