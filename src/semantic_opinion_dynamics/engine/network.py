from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx

from semantic_opinion_dynamics.contracts import NetworkSpec


class Network:
    def __init__(self, spec: NetworkSpec) -> None:
        self.spec = spec
        self.g = nx.DiGraph() if spec.directed else nx.Graph()
        for n in spec.nodes:
            self.g.add_node(n.id, **(n.model_dump(exclude={"id"}) or {}))
        for e in spec.edges:
            self.g.add_edge(e.source, e.target, weight=e.weight)

    def node_ids(self) -> List[str]:
        return list(self.g.nodes())

    def incoming_neighbors(self, node_id: str) -> List[Tuple[str, float]]:
        if isinstance(self.g, nx.DiGraph):
            nbs = list(self.g.predecessors(node_id))
            return [(n, float(self.g[n][node_id].get("weight", 1.0))) for n in nbs]
        nbs = list(self.g.neighbors(node_id))
        return [(n, float(self.g[node_id][n].get("weight", 1.0))) for n in nbs]

    def outgoing_neighbors(self, node_id: str) -> List[Tuple[str, float]]:
        if isinstance(self.g, nx.DiGraph):
            nbs = list(self.g.successors(node_id))
            return [(n, float(self.g[node_id][n].get("weight", 1.0))) for n in nbs]
        return self.incoming_neighbors(node_id)

    def to_networkx(self) -> nx.Graph:
        return self.g

    def edge_weights(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for u, v, data in self.g.edges(data=True):
            out.setdefault(u, {})[v] = float(data.get("weight", 1.0))
        return out
