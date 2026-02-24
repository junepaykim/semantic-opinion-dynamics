"""
Semantic Opinion Dynamics: main entry point.
Generate networks or run iterations via CLI arguments.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from input import generateNetwork, initNodes, loadNetwork, saveNetwork
from model import agentIterate, degrootIterate, hasConverged


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
        choices=["random", "small_world", "scale_free"],
        default="random",
        help="Graph type when generating: random, small_world, scale_free.",
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
        help="Iteration count. If omitted, run until convergence.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        metavar="E",
        help="Convergence threshold: stop when max|opinionScore change| < epsilon. Default: agent 1e-2, degroot 1e-6.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=10000,
        help="Max iterations in converge mode to prevent infinite loop (default: 10000).",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Run initNodes to fill persona and prompt after generating.",
    )
    return parser.parse_args()


def main():
    args = parseArgs()

    if args.generate:
        # Task: generate new network
        network = generateNetwork(
            nNodes=args.nodes,
            graphType=args.graph_type,
            outputName=args.name,
        )
        outputName = args.name
        if args.init:
            network = initNodes(network, outputName=outputName)
        else:
            saveNetwork(network, name=outputName)
        print(f"Generated network: {len(network.get('nodes', []))} nodes")
        return

    # Task: load network and run iteration
    if not args.name:
        print("Error: --name (-n) required when not generating. Specify which network from networks/ to use.")
        sys.exit(1)

    network = loadNetwork(args.name)
    iterateFn = agentIterate if args.model == "agent" else degrootIterate

    # Agent model LLM output is stochastic; use model-specific default epsilon
    epsilon = args.epsilon if args.epsilon is not None else (1e-2 if args.model == "agent" else 1e-6)

    # Create networks/{name}_slices/ to record graph state per iteration
    slicesDir = f"{args.name}_slices"
    saveNetwork(network, name=f"{slicesDir}/iter0")  # Initial state

    if args.iters is not None:
        # Fixed iteration count
        for i in range(args.iters):
            network = iterateFn(network, outputName=f"{slicesDir}/iter{i + 1}")
        print(f"Completed {args.iters} iteration(s), model: {args.model}. Slices saved to networks/{slicesDir}/.")
    else:
        # Run until convergence
        i = 0
        while i < args.max_iters:
            prevScores = {n["id"]: n["opinionScore"] for n in network["nodes"]}
            network = iterateFn(network, outputName=f"{slicesDir}/iter{i + 1}")
            i += 1
            if hasConverged(prevScores, network, epsilon=epsilon):
                print(f"Converged after {i} iteration(s), model: {args.model}. Slices saved to networks/{slicesDir}/.")
                break
        else:
            print(f"Reached max iterations {args.max_iters} without convergence. Slices saved to networks/{slicesDir}/.")


if __name__ == "__main__":
    main()
