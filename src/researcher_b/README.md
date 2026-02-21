# Researcher B — Metrics, Embeddings, and Visualization

Your job: treat the engine outputs as the **dataset**, then compute semantic metrics (SBERT embeddings, polarization) and create plots.

You do *not* need to modify the engine.

## What you consume

A completed run directory:

- `runs/<run_id>/steps/step_0000.json ... step_XXXX.json`
- `runs/<run_id>/network_used.json`
- `runs/<run_id>/personas_used.yaml`
- `runs/<run_id>/config_used.yaml`

Contracts (Pydantic):
- `semantic_opinion_dynamics.contracts.runlog.StepLog`
- `semantic_opinion_dynamics.contracts.opinion.OpinionState`

## Minimal loader snippet

```bash
PYTHONPATH=src uv run python - <<'PY'
import glob, json
from semantic_opinion_dynamics.contracts.runlog import StepLog

run_id = "<run_id>"
paths = sorted(glob.glob(f"runs/{run_id}/steps/step_*.json"))

steps = []
for p in paths:
    with open(p, "r", encoding="utf-8") as f:
        steps.append(StepLog.model_validate(json.load(f)))

print("loaded steps:", len(steps))
print("agents:", len(steps[0].agents))
PY
```

## What to implement

- **Embeddings**
  - SBERT / sentence-transformers (add dependency in `pyproject.toml` if needed)
  - Embed either:
    - `state.opinion_text` (recommended), or
    - a structured text concatenation of `stance/tone/key_arguments/opinion_text`
- **Metrics (examples)**
  - semantic variance over time (mean cosine distance to centroid)
  - polarization score (e.g., bimodality / cluster separation)
  - stance distribution over time (counts of stance enums)
- **Plots**
  - t-SNE / PCA scatter per timestep
  - time series of polarization metrics
  - heatmaps of pairwise distances

## Where to write outputs

Recommended conventions:
- save analysis artifacts under `runs/<run_id>/analysis/`
  - `metrics.csv`
  - `embeddings.npy`
  - `figures/*.png`

Deliverables for the team: clear plots + a short interpretation of what changed across timesteps and why.
