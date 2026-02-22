"""
Update nodes: call LLM to update opinionScore and prompt for each node.
"""


def updateNodeOpinion(node: dict) -> dict:
    """
    Call LLM API to update node's opinionScore and prompt in place, and return the updated content.
    Based on current persona, opinionScore, neighbor info, etc., LLM generates new opinion score and description. prompt at most 50 words.

    Args:
        node: Node dict with id, opinionScore, prompt, persona, neighbors, etc. opinionScore and prompt modified in place.

    Returns:
        Updated content, e.g. {"opinionScore": float, "prompt": str}
    """
    # TODO: implement
    raise NotImplementedError


def updateNodes(network: dict, outputName: str | None = None) -> dict:
    """
    Update nodes: call updateNodeOpinion to update opinionScore and prompt for each node.
    Based on current persona, neighbor info, etc., LLM generates new opinion score and description.
    Updates network in place and returns the updated graph.

    Args:
        network: dict with nodes (each node has id, opinionScore, prompt, persona, neighbors). Modified in place.
        outputName: If provided, save to networks/{outputName}.json

    Returns:
        Updated network (the same dict, with each node's opinionScore and prompt updated)
    """
    # TODO: Iterate nodes, call updateNodeOpinion(node), write returned updates to node. If outputName, call saveNetwork from input.networkGen.
    raise NotImplementedError
