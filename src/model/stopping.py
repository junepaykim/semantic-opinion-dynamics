"""
Stopping rule: determine whether graph iteration has converged.
Convergence when opinionScore change for all nodes is below threshold.
"""


def maxOpinionChange(prevScores: dict[str, float], currNetwork: dict) -> float:
    """
    Compute max opinionScore change across all nodes before/after iteration.

    Args:
        prevScores: {nodeId: opinionScore} before iteration
        currNetwork: network dict after iteration (with nodes)

    Returns:
        max |prev - curr| over all nodes
    """
    nodes = currNetwork.get("nodes", [])
    maxDiff = 0.0
    for n in nodes:
        nid = n["id"]
        curr = n["opinionScore"]
        prev = prevScores.get(nid, curr)
        maxDiff = max(maxDiff, abs(prev - curr))
    return maxDiff


def hasConverged(
    prevScores: dict[str, float],
    currNetwork: dict,
    epsilon: float = 1e-6,
) -> bool:
    """
    Check convergence: all nodes have opinionScore change < epsilon.

    Args:
        prevScores: {nodeId: opinionScore} before iteration
        currNetwork: network dict after iteration
        epsilon: convergence threshold (default 1e-6)

    Returns:
        True if converged
    """
    return maxOpinionChange(prevScores, currNetwork) < epsilon
