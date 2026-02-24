"""
Network generation and node initialization: build graph structure, assign prompt etc. to nodes.
"""

import json
from pathlib import Path
from typing import Literal

import networkx as nx

# Project root
root = Path(__file__).resolve().parent.parent.parent
networksDir = root / "networks"


def generateNetwork(
    nNodes: int = 20,
    graphType: Literal["random", "scale_free", "small_world"] = "random",
    outputName: str | None = None,
    **kwargs,
) -> dict:
    """
    Generate network, save as JSON to networks/ directory, and return the graph. Topic: cats vs dogs.

    Args:
        nNodes: Number of nodes
        graphType: Graph type. Implement three strategies:
          - "random": Erdős–Rényi random graph. Each edge exists with probability p. kwargs: p (default 0.1)
          - "scale_free": Barabási–Albert scale-free network. Preferential attachment. kwargs: m (edges per new node, default 2)
          - "small_world": Watts–Strogatz small-world network. Start from ring, rewire with probability p. kwargs: k (neighbors per node, default 4), p (rewire prob, default 0.3)
        outputName: If provided, save to networks/{outputName}.json; otherwise auto-name as Net1, Net2, ...
        **kwargs: Extra params per graph type (p, m, k, seed, etc.)

    Returns:
        Network dict (also saved as networks/Net{N}.json). Contains nodes only. Each node has id, opinionScore, prompt, persona, neighbors.
          - id: str, numeric string, e.g. "1"～"100"
          - opinionScore: float [0, 1], stance on topic. 0=prefer cats, 1=prefer dogs, 0.5=both with preference.
            Implement with normal distribution sampling in [0, 1].
          - prompt: str, initialized as empty, filled by initNodes via modelCall.
          - persona: str, initialized as empty, filled by initNodes via modelCall.
          - neighbors: dict, adjacent neighbors and edge weights, {neighborId: weight}. weight [0, 1], influence degree.
            Implement with normal distribution sampling in [0, 1].

    Example:
        {
            "nodes": [
                {
                    "id": "1", 
                    "opinionScore": 0.2, 
                    "prompt": "", 
                    "persona": "", 
                    "neighbors": {"2": 0.6}
                },
                {
                    "id": "2", 
                    "opinionScore": 0.8, 
                    "prompt": "", 
                    "persona": "", 
                    "neighbors": {"1": 0.6}
                }
            ]
        }
    """
    # TODO:


def initNodes(network: dict, outputName: str | None = None) -> dict:
    """
    Initialize nodes: call modelCall.generatePersona, modelCall.generateOpinionPrompt to generate persona and prompt for each node.
    Updates network in place and returns the updated graph.

    Args:
        network: dict from generateNetwork (nodes have id, opinionScore, prompt="", persona="", neighbors). Modified in place.
        outputName: If provided, save to networks/{outputName}.json

    Returns:
        Updated network (the same dict, with node["persona"] and node["prompt"] filled)
    """
    # TODO: Iterate nodes, call generatePersona(opinionScore), generateOpinionPrompt(opinionScore, persona), write to node["persona"], node["prompt"]
    raise NotImplementedError




def loadNetwork(name: str) -> dict:
    """
    Load network from networks/{name}.json.

    Args:
        name: Network name without .json extension, e.g. "Net1" or "myGraph"

    Returns:
        Network dict with nodes.
    """
    path = networksDir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Network not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def saveNetwork(network: dict, name: str | None = None) -> Path:
    """
    Save network as JSON to networks/ directory.
    If name is None, auto-name as Net1.json, Net2.json, ... incrementing.
    name may include subpath, e.g. "Net1_slices/iter1", which creates networks/Net1_slices/.
    """
    networksDir.mkdir(parents=True, exist_ok=True)
    if name is None:
        existing = list(networksDir.glob("Net*.json"))
        nums = []
        for p in existing:
            try:
                n = int(p.stem.replace("Net", ""))
                nums.append(n)
            except ValueError:
                pass
        nextNum = max(nums, default=0) + 1
        name = f"Net{nextNum}"
    path = networksDir / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(network, f, ensure_ascii=False, indent=2)
    return path
