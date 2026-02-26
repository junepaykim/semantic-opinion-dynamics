"""Network ops: generate, load, save, init nodes."""

import json
import re
from pathlib import Path
from typing import Literal
import networkx as nx
import numpy as np
from tqdm import tqdm

from . import modelCall

_ROOT = Path(__file__).resolve().parent.parent.parent
networksDir = _ROOT / "networks"
PRECISION = 6
GRAPH_TYPE_SUFFIX = {"random": "ER", "small_world": "SW", "scale_free": "SF", "karate_club": "KC"}


def getNextNetworkBasename() -> str:
    """Next auto basename (Net1, Net2, ...) from existing files."""
    networksDir.mkdir(parents=True, exist_ok=True)
    existing = list(networksDir.glob("Net*.json"))
    nums = []
    for p in existing:
        m = re.match(r"^Net(\d+)", p.stem)
        if m:
            nums.append(int(m.group(1)))
    nextNum = max(nums, default=0) + 1
    return f"Net{nextNum}"


def generateNetwork(
    nNodes: int = 20,
    graphType: Literal["random", "scale_free", "small_world", "karate_club"] = "random",
    **kwargs,
) -> dict:
    """Generate graph structure. Returns network dict with nodes (id, opinionScore, prompt, persona, neighbors)."""
    seed = kwargs.get("seed")

    if graphType == "random":
        p = kwargs.get("p", 0.1)
        G = nx.erdos_renyi_graph(n=nNodes, p=p, seed=seed, directed=False)
    elif graphType == "scale_free":
        m = min(kwargs.get("m", 2), nNodes - 1)
        G = nx.barabasi_albert_graph(n=nNodes, m=m, seed=seed)
    elif graphType == "small_world":
        k, p = kwargs.get("k", 4), kwargs.get("p", 0.3)
        if k % 2:
            k += 1
        if k >= nNodes:
            k = (nNodes // 2) * 2
        G = nx.watts_strogatz_graph(n=nNodes, k=k, p=p, seed=seed)
    elif graphType == "karate_club":
        G = nx.karate_club_graph()
        nNodes = 34
    else:
        raise ValueError(f"Unsupported graph type: {graphType}")

    scores = np.random.normal(loc=0.5, scale=0.2, size=nNodes)
    scores = np.clip(scores, 0.0, 1.0)

    edge_weights = {}
    for u, v in G.edges():
        key = (min(u, v), max(u, v))
        if key not in edge_weights:
            w = np.random.normal(loc=0.5, scale=0.2)
            w = np.clip(w, 0.0, 1.0)
            edge_weights[key] = round(float(w), PRECISION)

    nodes = []
    for node in tqdm(range(nNodes), desc="Generating nodes", unit="node"):
        node_id = str(node + 1)
        neighbors = {}
        for neighbor in G.neighbors(node):
            neighbor_id = str(neighbor + 1)
            key = (min(node, neighbor), max(node, neighbor))
            neighbors[neighbor_id] = edge_weights[key]
        node_entry = {
            "id": node_id,
            "opinionScore": round(float(scores[node]), PRECISION),
            "prompt": "",
            "persona": "",
            "neighbors": neighbors,
        }
        nodes.append(node_entry)

    network = {"nodes": nodes}

    return network


def initNodes(network: dict, outputName: str | None = None) -> dict:
    """Init nodes: generate persona and prompt via LLM."""
    for node in tqdm(network["nodes"], desc="Init nodes", unit="node"):
        score = node["opinionScore"]
        node["persona"] = modelCall.generatePersona(score)
        node["prompt"] = modelCall.generateOpinionPrompt(score, node["persona"])
    saveNetwork(network, outputName)
    return network


def loadNetwork(name: str) -> dict:
    """Load network from networks/{name}.json."""
    path = networksDir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Network not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        network = json.load(f)
    _round(network)
    return network


def _round(network: dict) -> None:
    """Round scores and weights in place."""
    for node in network.get("nodes", []):
        if "opinionScore" in node:
            node["opinionScore"] = round(float(node["opinionScore"]), PRECISION)
        for nid, w in list(node.get("neighbors", {}).items()):
            node["neighbors"][nid] = round(float(w), PRECISION)


def saveNetwork(network: dict, name: str | None = None) -> Path:
    """Save to networks/{name}.json. Auto-name Net1, Net2, ... if name is None."""
    networksDir.mkdir(parents=True, exist_ok=True)
    if name is None:
        name = getNextNetworkBasename()
    path = networksDir / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _round(network)
    with path.open("w", encoding="utf-8") as f:
        json.dump(network, f, ensure_ascii=False, indent=2)
    return path
