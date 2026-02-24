"""
Agent-based graph iteration: update nodes via LLM.
Each node's opinionScore and prompt are generated and updated by LLM from persona and neighbor info.
"""


def updateNode(node: dict) -> dict:
    """
    Call LLM API to update node's opinionScore and prompt in place; return updated content.
    LLM generates new opinion score and opinion description from persona, opinionScore, neighbor info, etc. prompt ≤ 50 words.

    Args:
        node: node dict with id, opinionScore, prompt, persona, neighbors. opinionScore and prompt modified in place.

    Returns:
        Updated content, e.g. {"opinionScore": float, "prompt": str}
    """
    # TODO: implement
    raise NotImplementedError


def agentIterate(network: dict, outputName: str | None = None) -> dict:
    """
    Run one Agent graph iteration: call updateNode for each node to update opinionScore and prompt.
    LLM generates new opinion score and description from persona, neighbor info, etc.
    Updates network in place and returns the updated graph.

    Args:
        network: dict with nodes (each has id, opinionScore, prompt, persona, neighbors). Modified in place.
        outputName: if provided, save to networks/{outputName}.json

    Returns:
        Updated network (same dict, opinionScore and prompt updated for all nodes)
    """
    # TODO: iterate nodes, call updateNode(node), write returned content to node. If outputName, call input.networkGen saveNetwork.
    raise NotImplementedError
