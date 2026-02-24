"""
Network generation and node initialization: build graph structure, assign prompt etc. to nodes.
"""

import json
from pathlib import Path
from typing import Literal
import random
import networkx as nx
import numpy as np

import modelCall

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
    Generate network, save as JSON to networks/ directory, and return the graph. Topic: remote work vs work from office.

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
    # Set seed if provided
    seed = kwargs.get("seed", None)

    # Generate network graph based on type
    if graphType == "random":   # Erdős–Rényi random graph
        p = kwargs.get("p", 0.1)
        G = nx.erdos_renyi_graph(n=nNodes, p=p, seed=seed, directed=False)
    elif graphType == "scale_free": # Barabási–Albert scale-free network
        m = kwargs.get("m", 2)
        m = min(m, nNodes - 1)  # prevent invalid m
        G = nx.barabasi_albert_graph(n=nNodes, m=m, seed=seed, initial_graph=None)
    elif graphType == "small_world":    # Watts–Strogatz small-world network
        k = kwargs.get("k", 4)
        p = kwargs.get("p", 0.3)
        if k % 2 != 0:
            k += 1
        if k >= nNodes:
            k = (nNodes // 2) * 2   # prevent invalid k
        G = nx.watts_strogatz_graph(n=nNodes, k=k, p=p, seed=seed)
    elif graphType == "karate_club":    # Zachary's Karate Club network
        G = nx.karate_club_graph()
        nNodes = 34  # override nNodes to match graph
    else:
        raise ValueError(f"Unsupported graph type: {graphType}")

    # Generate Opinion Scores
    scores = np.random.normal(loc=0.5, scale=0.2, size=nNodes)
    scores = np.clip(scores, 0.0, 1.0)

    # Generate Edge Weights
    edge_weights = {}
    for u, v in G.edges():
        key = (min(u, v), max(u, v))
        if key not in edge_weights:
            w = np.random.normal(loc=0.5, scale=0.2)
            w = np.clip(w, 0.0, 1.0)
            edge_weights[key] = w

    # Generate the final network dict
    nodes = []
    for node in range(nNodes):
        node_id = str(node + 1)
        neighbors = {}
        for neighbor in G.neighbors(node):
            neighbor_id = str(neighbor + 1)
            key = (min(node, neighbor), max(node, neighbor))
            neighbors[neighbor_id] = edge_weights[key]
        node_entry = {
            "id": node_id,
            "opinionScore": float(scores[node]),
            "prompt": "",
            "persona": "",
            "neighbors": neighbors,
        }
        nodes.append(node_entry)

    network = {"nodes": nodes}

    # Save network to file
    path = saveNetwork(network)

    # Initialize nodes with persona and prompt
    network = initNodes(network, output=path.stem)

    return network

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
    # Iterate nodes
    for node in network["nodes"]:
        opinion = node["opinionScore"]
        
        # Get persona
        persona = modelCall.generatePersona(opinion)
        node["persona"] = persona
        
        # Get Original Opinion based on persona
        prompt = modelCall.generateOpinionPrompt(opinion, persona)
        node["prompt"] = prompt

    # Save initialized network
    saveNetwork(network, outputName)

    return network


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
