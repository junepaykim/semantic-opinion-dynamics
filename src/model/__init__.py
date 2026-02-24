"""Model: LLM-based Agent graph iteration (agentModel) and classic DeGroot iteration (baseline)."""

from .agentModel import agentIterate, updateNode
from .baseline import degrootIterate
from .stopping import hasConverged, maxOpinionChange

__all__ = ["agentIterate", "updateNode", "degrootIterate", "hasConverged", "maxOpinionChange"]
