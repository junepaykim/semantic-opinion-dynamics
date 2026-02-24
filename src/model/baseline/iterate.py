"""
Classic DeGroot graph iteration: weighted average update.
Each node's opinionScore is updated to the weighted average of neighbors' opinionScores.
"""

from input import saveNetwork


def degrootIterate(network: dict, outputName: str | None = None) -> dict:
    """
    DeGroot iteration: update each node's opinionScore to neighbor weighted average.
    x_i(t+1) = sum_{j in N(i)} w_ij * x_j(t), row-normalized weights.
    Only opinionScore is updated; prompt, persona unchanged.

    Args:
        network: dict with nodes (each has id, opinionScore, neighbors). Modified in place.
        outputName: if provided, save to networks/{outputName}.json

    Returns:
        Updated network (same dict, opinionScore updated for all nodes)
    """
    nodes = network.get("nodes", [])
    id_to_node = {n["id"]: n for n in nodes}

    # Compute all new values first to avoid reading modified neighbor values during in-place update
    new_scores: dict[str, float] = {}
    for node in nodes:
        nid = node["id"]
        neighbors = node.get("neighbors", {})
        if not neighbors:
            new_scores[nid] = max(0.0, min(1.0, node["opinionScore"]))
            continue
        total_weight = 0.0
        weighted_sum = 0.0
        for jid, wij in neighbors.items():
            j_node = id_to_node.get(jid)
            if j_node is not None:
                total_weight += wij
                weighted_sum += wij * j_node["opinionScore"]
        score = weighted_sum / total_weight if total_weight > 0 else node["opinionScore"]
        new_scores[nid] = max(0.0, min(1.0, score))

    for node in nodes:
        node["opinionScore"] = new_scores[node["id"]]

    if outputName:
        saveNetwork(network, name=outputName)

    return network
