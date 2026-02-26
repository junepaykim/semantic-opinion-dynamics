"""Input: network ops, model call."""

from .modelCall import generateOpinionPrompt, generatePersona
from .networkOps import (
    GRAPH_TYPE_SUFFIX,
    generateNetwork,
    getNextNetworkBasename,
    initNodes,
    loadNetwork,
    saveNetwork,
)
from model.agentModel import agentIterate, updateNode

__all__ = [
    "generateNetwork",
    "getNextNetworkBasename",
    "GRAPH_TYPE_SUFFIX",
    "initNodes",
    "loadNetwork",
    "saveNetwork",
    "agentIterate",
    "updateNode",
    "generateOpinionPrompt",
    "generatePersona",
]