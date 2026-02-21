from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .enums import Stance, Tone


class OpinionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stance: Stance
    confidence: float = Field(ge=0.0, le=1.0)
    tone: Tone
    key_arguments: List[str] = Field(default_factory=list, max_length=8)
    opinion_text: str = Field(min_length=1, max_length=2000)


class OpinionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stance: Stance
    confidence: float = Field(ge=0.0, le=1.0)
    tone: Tone
    key_arguments: List[str] = Field(default_factory=list, max_length=8)
    updated_opinion: str = Field(min_length=1, max_length=2000)
    reply_to_neighbors: Optional[bool] = None
