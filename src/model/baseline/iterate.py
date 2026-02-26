"""DeGroot iteration: weighted average of neighbor opinions."""

from tqdm import tqdm

from input import saveNetwork

PRECISION = 6


def degrootIterate(network: dict, outputName: str | None = None) -> dict:
    """One DeGroot step: x_i = weighted avg of neighbors. Optionally save."""
    nodes = network.get("nodes", [])
    id_to_node = {n["id"]: n for n in nodes}

    # Compute all new values first to avoid reading modified neighbor values during in-place update
    new_scores: dict[str, float] = {}
    for node in tqdm(nodes, desc="DeGroot iteration", unit="node"):
        nid = node["id"]
        neighbors = node.get("neighbors", {})
        if not neighbors:
            new_scores[nid] = round(max(0.0, min(1.0, node["opinionScore"])), PRECISION)
            continue
        total_weight = 0.0
        weighted_sum = 0.0
        for jid, wij in neighbors.items():
            j_node = id_to_node.get(jid)
            if j_node is not None:
                total_weight += wij
                weighted_sum += wij * j_node["opinionScore"]
        score = weighted_sum / total_weight if total_weight > 0 else node["opinionScore"]
        new_scores[nid] = round(max(0.0, min(1.0, score)), PRECISION)

    for node in nodes:
        node["opinionScore"] = new_scores[node["id"]]

    if outputName:
        saveNetwork(network, outputName)

    return network
