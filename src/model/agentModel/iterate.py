"""Agent iteration: update nodes via LLM."""

from tqdm import tqdm

from input import modelCall, saveNetwork

PRECISION = 6


def updateNode(node: dict, id_to_node: dict[str, dict]) -> dict:
    """Update node opinion and prompt via LLM from persona and neighbor info."""
    neighbors = node.get("neighbors", {})
    neighbor_info: list[tuple[str, float]] = []
    for jid, wij in neighbors.items():
        j_node = id_to_node.get(jid)
        if j_node is not None:
            j_prompt = j_node.get("prompt", "")
            neighbor_info.append((j_prompt, wij))

    score, promptText = modelCall.updateNodeOpinion(
        persona=node.get("persona", ""),
        current_score=float(node.get("opinionScore", 0.5)),
        current_prompt=node.get("prompt", ""),
        neighbor_info=neighbor_info,
    )
    return {"opinionScore": score, "prompt": promptText}


def agentIterate(network: dict, outputName: str | None = None) -> dict:
    """One agent iteration: update all nodes via LLM, optionally save."""
    nodes = network.get("nodes", [])
    id_to_node = {n["id"]: n for n in nodes}
    for node in tqdm(nodes, desc="Agent iter", unit="node"):
        u = updateNode(node, id_to_node)
        node["opinionScore"] = round(float(u.get("opinionScore", node["opinionScore"])), PRECISION)
        node["prompt"] = u.get("prompt", node["prompt"])
    if outputName:
        saveNetwork(network, outputName)
    return network
