"""Input: network generation and node initialization."""

from .modelCall import generateOpinionPrompt, generatePersona
from .networkGen import generateNetwork, initNodes, loadNetwork, saveNetwork

# Re-export from model.agentModel for convenience
from ..model.agentModel import agentIterate, updateNode

__all__ = ["generateNetwork", "initNodes", "loadNetwork", "saveNetwork", "agentIterate", "updateNode", "generateOpinionPrompt", "generatePersona"]


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