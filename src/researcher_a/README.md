# Researcher A — Scenario & Prompt Inputs

Your job: define **what the simulation is about** (personas + network + configs).
You do *not* need to change the engine.

## What you produce

### 1) Network topology (JSON)

Put graphs under `networks/` (recommended) and pass the path to the engine.

- Contract: `semantic_opinion_dynamics.contracts.network.NetworkSpec`
- Semantics:
  - `directed: true` means edges are directional.
  - `source -> target` means **source influences target**.
  - The engine uses **incoming neighbors** of each node as the messages it “reads”.

Minimal example:

```json
{
  "directed": true,
  "nodes": [{"id": "mgr_1"}, {"id": "eng_1"}],
  "edges": [{"source": "mgr_1", "target": "eng_1", "weight": 1.0}]
}
```

### 2) Personas + initial opinions (YAML)

Create `personas*.yaml` at repo root (recommended) and pass the file path to the engine.

- Contract: `semantic_opinion_dynamics.contracts.persona.PersonasSpec`
- Each key under `agents:` is the **agent_id** and must match a node id in the network JSON.

Notes:
- `initial_opinion` must match `OpinionState` (stance/tone/confidence/text).
- Extra fields are allowed on each agent (e.g., you can add `frozen: true` for bots).

### 3) Config variants (YAML)

Create `config*.yaml` files to run multiple scenarios quickly.

- Contract: `semantic_opinion_dynamics.contracts.config.RunConfig`
- Typical knobs:
  - `experiment.steps`
  - `experiment.max_neighbors_in_prompt`
  - `prompts.behavior_rules` (global behavior constraints)
  - `llm.provider` (keep as `dummy` unless Architect adds a real provider)

## Validate your artifacts (recommended)

```bash
PYTHONPATH=src uv run python - <<'PY'
import json, yaml
from semantic_opinion_dynamics.contracts.network import NetworkSpec
from semantic_opinion_dynamics.contracts.persona import PersonasSpec

with open("networks/example_org.json") as f:
    NetworkSpec.model_validate(json.load(f))

with open("personas.yaml") as f:
    PersonasSpec.model_validate(yaml.safe_load(f))

print("OK: network + personas are valid")
PY
```

## Run the engine with your inputs

```bash
PYTHONPATH=src uv run python -m semantic_opinion_dynamics.cli run \
  --config config.yaml \
  --network networks/<your_graph>.json \
  --personas personas.yaml
```

Deliverables for the team: a small set of named scenario files (network + personas + config) that reliably produce interesting dynamics.
