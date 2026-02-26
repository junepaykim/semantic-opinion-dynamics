# Semantic Opinion Dynamics — Technical Documentation

## 0. Installation

**Requirements:** Python 3.10+

```bash
# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

**LLM backends:**
- **OpenAI** (when online): put your API key in `api_key.txt` (project root), or set `OPENAI_API_KEY` env var. Uses `gpt-3.5-turbo`. `api_key.txt` is in `.gitignore` and will not be uploaded.
- **Ollama** (fallback when offline): [Ollama](https://ollama.ai/) with a model (e.g. `ollama pull qwen3:4b`).

---

## 1. Project Overview

This project implements **Semantic Opinion Dynamics** simulation: placing LLM agents on social networks to study opinion evolution on the topic "cats vs dogs". Nodes have `opinionScore` (0=prefer cats, 1=prefer dogs), updated iteratively via LLM or the classic DeGroot model.

---

## 2. Project Structure

```
semantic-opinion-dynamics/
├── main.py                 # Main entry point, CLI dispatch
├── requirements.txt       # Dependencies
├── src/
│   ├── input/              # Input module: network operations
│   │   ├── networkOps.py  # generateNetwork, loadNetwork, saveNetwork, initNodes
│   │   └── modelCall.py   # LLM API calls (persona, prompt)
│   ├── model/             # Model module: graph iteration
│   │   ├── agentModel/    # LLM-based Agent iteration
│   │   │   └── iterate.py
│   │   └── baseline/      # Classic DeGroot iteration
│   │       └── iterate.py
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
    │   → initNodes → save      │       │ saveNetwork(iter0)         │
    │   (graph + persona+prompt)│       │ loop: iterate → save      │
    └───────────────────────────┘       │ --iters required          │
                                        └───────────────────────────┘
```

### 4.2 Task 1: Generate Network (networkOps)

```
generateNetwork(nNodes, graphType)
    │
    ├── Build graph (random / small_world / scale_free)
    ├── For each node: id, opinionScore, neighbors
    └── prompt, persona initialized as ""

initNodes(network, outputName)
    │
    ├── For each node: generatePersona(score)
    ├── For each node: generateOpinionPrompt(score, persona)
    └── saveNetwork → networks/{outputName}.json
```

### 4.3 Task 2: Iteration Update

```
loadNetwork(name)
    │
saveNetwork(iter0)  # Initial state
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Run --iters iterations:                                          │
│   network = iterateFn(network, outputName=slices/iter{i})       │
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

---

## 5. CLI Arguments

| Argument | Description | Default |
|----------|--------------|---------|
| `-g` | Generate new network | - |
| `-t` | Graph type: random / small_world / scale_free | random |
| `-n` | Network name (output when generating, load when running) | - |
| `--nodes` | Node count when generating | 20 |
| `--model` | Iteration model: agent / degroot | agent |
| `--iters` | Number of iterations (required when running) | - |

---

## 6. Usage Examples

```bash
# Generate random graph, 20 nodes, persona and prompt via LLM, save as myNet
python main.py -g -t random -n myNet

# Generate small-world graph (graph + persona + prompt)
python main.py -g -t small_world -n swNet

# Load myNet, run 5 agent iterations (--iters required)
python main.py -n myNet --iters 5

# Load myNet, degroot model, 20 iterations
python main.py -n myNet --model degroot --iters 20
```

---

## 7. Module Dependencies

```
main.py
  ├── input.generateNetwork, initNodes, loadNetwork, saveNetwork
  └── model.agentIterate, degrootIterate

input/networkOps
  └── generateNetwork, initNodes, loadNetwork, saveNetwork

input/modelCall
  └── generatePersona, generateOpinionPrompt

model/agentModel/iterate
  ├── updateNode  # Calls modelCall internally, TODO
  └── input.saveNetwork

model/baseline/iterate
  └── input.saveNetwork

```

---

## 8. Implementation Status

| Module | Status |
|--------|--------|
| main.py | ✅ Done |
| networkOps: generateNetwork | ⏳ TODO |
| networkOps: initNodes | ⏳ TODO |
| networkOps: loadNetwork, saveNetwork | ✅ Done |
| modelCall: generatePersona, generateOpinionPrompt | ✅ Done |
| agentModel: updateNode, agentIterate | ⏳ TODO |
| baseline: degrootIterate | ✅ Done |
