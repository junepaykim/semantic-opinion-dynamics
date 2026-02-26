"""Model: agent (LLM) and degroot iteration."""

from .agentModel import agentIterate, updateNode
from .baseline import degrootIterate

__all__ = ["agentIterate", "updateNode", "degrootIterate"]
