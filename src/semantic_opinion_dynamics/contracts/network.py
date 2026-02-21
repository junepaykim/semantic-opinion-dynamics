from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class NodeSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1, max_length=200)
    type: Optional[str] = Field(default=None, max_length=50)
    meta: Optional[Dict[str, Any]] = None


class EdgeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1, max_length=200)
    target: str = Field(min_length=1, max_length=200)
    weight: float = Field(default=1.0, gt=0.0, le=100.0)


class NetworkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directed: bool = True
    nodes: List[NodeSpec]
    edges: List[EdgeSpec]
