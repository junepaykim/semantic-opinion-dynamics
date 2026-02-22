"""Input: network generation and node initialization."""

from .modelCall import generateOpinionPrompt, generatePersona
from .networkGen import generateNetwork, initNodes, saveNetwork

# Re-export from model.agentModel for convenience
from ..model.agentModel import updateNodeOpinion, updateNodes

__all__ = ["generateNetwork", "initNodes", "saveNetwork", "updateNodes", "generateOpinionPrompt", "generatePersona", "updateNodeOpinion"]


'''
generateNetwork
    |
    +---> saveNetwork
    |
    +---> initNodes
              |
              +---> generatePersona
              +---> generateOpinionPrompt
              +---> saveNetwork
'''