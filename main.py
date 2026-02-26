"""Semantic Opinion Dynamics: CLI entry point."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from input import (
    GRAPH_TYPE_SUFFIX,
    generateNetwork,
    getNextNetworkBasename,
    initNodes,
    loadNetwork,
    saveNetwork,
)
from model import agentIterate, degrootIterate


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Semantic Opinion Dynamics: generate networks or run iterations."
    )
    parser.add_argument(
        "-g",
        "--generate",
        action="store_true",
        help="Generate a new network and save to networks/.",
    )
    parser.add_argument(
        "-t",
        "--graph-type",
        choices=["random", "small_world", "scale_free", "karate_club"],
        default="random",
        help="Graph type when generating: random, small_world, scale_free, karate_club.",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="Network name in networks/. When generating: output name; when running: initial network to load.",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=20,
        help="Number of nodes when generating (default: 20).",
    )
    parser.add_argument(
        "--model",
        choices=["agent", "degroot"],
        default="agent",
        help="Iteration model: agent (LLM) or degroot (default: agent).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        metavar="N",
        help="Number of iterations (required when running).",
    )
    return parser.parse_args()


def main():
    args = parseArgs()

    if args.generate:
        network = generateNetwork(nNodes=args.nodes, graphType=args.graph_type)
        base = args.name or getNextNetworkBasename()
        out = f"{base}_{GRAPH_TYPE_SUFFIX[args.graph_type]}"
        initNodes(network, out)
        print(f"Generated {len(network['nodes'])} nodes -> networks/{out}.json")
        return

    if not args.name:
        print("Error: --name (-n) required. Specify network from networks/.")
        sys.exit(1)
    if args.iters is None:
        print("Error: --iters required when running. Specify number of iterations.")
        sys.exit(1)

    network = loadNetwork(args.name)
    iterateFn = agentIterate if args.model == "agent" else degrootIterate
    slicesDir = f"{args.name}_{args.model}_slices"
    saveNetwork(network, f"{slicesDir}/iter0")

    def maxOpinionChange(prevScores, net):
        return max(
            (abs(prevScores.get(n["id"], n["opinionScore"]) - n["opinionScore"]) for n in net.get("nodes", [])),
            default=0.0,
        )

    for i in range(1, args.iters + 1):
        prevScores = {n["id"]: n["opinionScore"] for n in network["nodes"]}
        network = iterateFn(network, f"{slicesDir}/iter{i}")
        maxDiff = maxOpinionChange(prevScores, network)
        print(f"iter{i}: maxDiff={maxDiff:.6f}")
    print(f"Completed {args.iters} iterations, model={args.model}. Slices: networks/{slicesDir}/")


if __name__ == "__main__":
    main()
