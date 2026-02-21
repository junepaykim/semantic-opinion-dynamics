# Semantic Opinion Dynamics - LLM Agents on Networks

This repo simulates **text-based opinion dynamics** on a small organizational network.
Each node is an agent with a persona + opinion text that updates over discrete time steps after “reading” neighbor messages.

## Architecture

- `src/semantic_opinion_dynamics/`: **engine**
  - `contracts/`: Pydantic **data contracts** (inputs + outputs)
  - `engine/`: Agent / Network / Simulation (synchronous update)
  - `llm/`: LLM client interface + **dummy** provider (real API not implemented)
  - `io/`: YAML/JSON loaders + run persistence
- `src/researcher_a/`: scenario inputs (personas + graphs + configs)
- `src/researcher_b/`: post-hoc analysis (embeddings, metrics, plots)
- `src/researcher_c/`: baselines + experimental comparisons (e.g., DeGroot)

## Data flow

1. **Inputs**
   - `config.yaml` → validated as `RunConfig`
   - `networks/<name>.json` → validated as `NetworkSpec`
   - `personas*.yaml` → validated as `PersonasSpec` (persona + initial `OpinionState`)
2. **Simulation**
   - Discrete timesteps `t = 0..T`
   - **Synchronous update:** all agents read the `t` snapshot and update to `t+1` together.
   - Neighbors are **incoming** edges (for directed graphs): `source -> target` means *source influences target*.
3. **Outputs**
   - `runs/<run_id>/steps/step_0000.json ...` (validated as `StepLog`)
   - plus `config_used.yaml`, `network_used.json`, `personas_used.yaml` snapshots

## Quickstart
Initial dev environment setup:
```bash 
python3 bootstrap.py
```
To run the code:
```bash
make init
make run
```

Run a different scenario:

```bash
PYTHONPATH=src uv run python -m semantic_opinion_dynamics.cli run \
  --config config.yaml \
  --network networks/example_org.json \
  --personas personas.yaml
```

## What to change most often

- `config.yaml`
  - `experiment.steps`: number of timesteps
  - `experiment.max_neighbors_in_prompt`: cap neighbor messages per agent
  - `experiment.store_prompts`: store prompts in step logs (debug / audit)
- `networks/*.json`: topology + weights (Researcher A)
- `personas*.yaml`: personas + initial opinions (Researcher A)

## Team workflow

- **Researcher A** produces `networks/*.json`, `personas*.yaml`, config variants.
- **Engine (Architect)** runs simulations and guarantees output stability via contracts in `contracts/`.
- **Researcher B** reads `runs/<run_id>/steps/*.json` and computes embeddings/metrics/plots.
- **Researcher C** implements classical baselines (e.g., DeGroot) and compares trajectories.

See per-role notes:
- `src/researcher_a/README.md`
- `src/researcher_b/README.md`
- `src/researcher_c/README.md`
