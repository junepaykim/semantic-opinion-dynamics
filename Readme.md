# Semantic Opinion Dynamics — Technical Documentation

## 1. Project Overview

This project implements **Semantic Opinion Dynamics** simulation: placing LLM agents on social networks to study opinion evolution on the topic "cats vs dogs". Nodes have `opinionScore` (0=prefer cats, 1=prefer dogs), updated iteratively via LLM or the classic DeGroot model.

---

## 2. Project Structure

```
semantic-opinion-dynamics/
├── main.py                 # Main entry point, CLI dispatch
├── src/
│   ├── input/              # Input module: network generation and node initialization
│   │   ├── networkGen.py  # Graph generation, load, save
│   │   └── modelCall.py   # LLM API calls (persona, prompt)
│   ├── model/             # Model module: graph iteration and stopping rules
│   │   ├── agentModel/    # LLM-based Agent iteration
│   │   │   └── iterate.py
│   │   ├── baseline/      # Classic DeGroot iteration
│   │   │   └── iterate.py
│   │   └── stopping.py   # Convergence check
│   └── visualization/    # Visualization (planned)
├── networks/              # Network JSON output directory
│   └── {name}_slices/    # Graph state snapshot per iteration
└── technical-documentation.md
```

---

## 3. Data Format

### 3.1 Network Structure (JSON)

```json
{
  "nodes": [
    {
      "id": "1",
      "opinionScore": 0.2,
      "prompt": "",
      "persona": "",
      "neighbors": {"2": 0.6, "3": 0.4}
    }
  ]
}
```

| Field | Type | Description |
|-------|------|--------------|
| `id` | str | Node ID |
| `opinionScore` | float [0,1] | Opinion score, 0=prefer cats, 1=prefer dogs |
| `prompt` | str | Opinion description (≤50 words), LLM-generated |
| `persona` | str | Persona tags (≤10 words), LLM-generated |
| `neighbors` | dict | `{neighborId: edgeWeight}`, weight ∈ [0,1] |

### 3.2 Iteration Slices Directory

When running iterations, each state is saved under `networks/{name}_slices/`:

```
networks/myNet_slices/
├── iter0.json   # Initial state
├── iter1.json   # After 1st iteration
├── iter2.json   # After 2nd iteration
└── ...
```

---

## 4. Workflow

### 4.1 Overall Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         main.py Main Entry                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            ┌───────────────┐               ┌───────────────┐
            │  -g Generate  │               │  Iterate Mode │
            └───────────────┘               └───────────────┘
                    │                               │
                    ▼                               ▼
    ┌───────────────────────────┐       ┌───────────────────────────┐
    │ generateNetwork           │       │ loadNetwork(args.name)     │
    │   → saveNetwork           │       │ saveNetwork(iter0)         │
    │   → [initNodes] (optional)│       │ loop: iterate → save      │
    └───────────────────────────┘       │ stop: --iters or converge │
                                        └───────────────────────────┘
```

### 4.2 Task 1: Generate Network

```
generateNetwork(nNodes, graphType, outputName)
    │
    ├── Build graph (random / small_world / scale_free)
    ├── For each node: id, opinionScore, neighbors
    ├── prompt, persona initialized as ""
    └── saveNetwork → networks/{outputName}.json

[if --init]
initNodes(network)
    │
    ├── For each node: generatePersona(opinionScore)
    ├── For each node: generateOpinionPrompt(opinionScore, persona)
    └── saveNetwork
```

### 4.3 Task 2: Iteration Update

```
loadNetwork(name)
    │
saveNetwork(iter0)  # Initial state
    │
┌───┴───────────────────────────────────────────────────────────┐
│ If --iters given: fixed N iterations                           │
│ If not: run until converge (max|ΔopinionScore| < epsilon) or --max-iters │
└───┬───────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Each iteration:                                                  │
│   prevScores = {id: opinionScore}  # Snapshot before iteration   │
│   network = iterateFn(network, outputName=slices/iter{i+1})     │
│   [converge mode] if hasConverged(prevScores, network, epsilon): break │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 Iteration Models

#### Agent Model (agentIterate)

```
agentIterate(network)
    │
    └── For each node: updateNode(node)
            │
            ├── Call LLM with persona, neighbor prompts, etc.
            ├── Generate new opinionScore, prompt (≤50 words)
            └── Update node in place
    │
    └── saveNetwork(outputName)
```

#### DeGroot Model (degrootIterate)

```
degrootIterate(network)
    │
    ├── For each node i:
    │     x_i(t+1) = Σ w_ij * x_j(t) / Σ w_ij  (neighbor weighted avg)
    │     opinionScore clamped to [0, 1]
    │
    └── saveNetwork(outputName)
```

### 4.5 Stopping Rule

```
hasConverged(prevScores, currNetwork, epsilon)
    │
    └── maxOpinionChange(prevScores, currNetwork) < epsilon
            │
            └── max |prev[i] - curr[i]|  over all nodes
```

| Model | Default epsilon | Note |
|-------|-----------------|------|
| agent | 1e-2 | LLM output is stochastic, hard to reach 1e-6 |
| degroot | 1e-6 | Deterministic update, can converge to smaller change |

---

## 5. CLI Arguments

| Argument | Description | Default |
|----------|--------------|---------|
| `-g` | Generate new network | - |
| `-t` | Graph type: random / small_world / scale_free | random |
| `-n` | Network name (output when generating, load when running) | - |
| `--nodes` | Node count when generating | 20 |
| `--model` | Iteration model: agent / degroot | agent |
| `--iters` | Iteration count; if omitted, run until converge | None |
| `--epsilon` | Convergence threshold | agent: 1e-2, degroot: 1e-6 |
| `--max-iters` | Max iterations in converge mode | 10000 |
| `--init` | Run initNodes after generating | - |

---

## 6. Usage Examples

```bash
# Generate random graph, 20 nodes, save as myNet
python main.py -g -t random -n myNet

# Generate small-world graph and init persona/prompt
python main.py -g -t small_world -n swNet --init

# Load myNet, fixed 5 agent iterations
python main.py -n myNet --iters 5

# Load myNet, run until converge (agent default epsilon=1e-2)
python main.py -n myNet

# Load myNet, degroot model, run until converge
python main.py -n myNet --model degroot

# Custom convergence threshold
python main.py -n myNet --epsilon 0.05 --max-iters 500
```

---

## 7. Module Dependencies

```
main.py
  ├── input.generateNetwork, initNodes, loadNetwork, saveNetwork
  └── model.agentIterate, degrootIterate, hasConverged

input/networkGen
  └── (generateNetwork calls saveNetwork internally)

input/modelCall
  └── generatePersona, generateOpinionPrompt  # Used by initNodes, agentModel

model/agentModel/iterate
  ├── updateNode  # Calls modelCall internally, TODO
  └── input.saveNetwork

model/baseline/iterate
  └── input.saveNetwork

model/stopping
  └── maxOpinionChange, hasConverged
```

---

## 8. Implementation Status

| Module | Status |
|--------|--------|
| main.py | ✅ Done |
| networkGen: generateNetwork | ⏳ TODO |
| networkGen: initNodes | ⏳ TODO |
| networkGen: loadNetwork, saveNetwork | ✅ Done |
| modelCall: generatePersona, generateOpinionPrompt | ⏳ TODO |
| agentModel: updateNode, agentIterate | ⏳ TODO |
| baseline: degrootIterate | ✅ Done |
| stopping: hasConverged, maxOpinionChange | ✅ Done |
