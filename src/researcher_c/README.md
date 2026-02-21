# Researcher C — Baselines & Experimental Comparisons

Your job: implement **classical baselines** (e.g., DeGroot weighted averaging) and design controlled experiments to compare against the LLM-agent simulation.

You do *not* need to modify the engine (but you may add scripts under this folder).

## What you consume

From a run directory (canonical snapshots):
- `runs/<run_id>/network_used.json` (graph + weights)
- `runs/<run_id>/personas_used.yaml` (initial opinions)
- `runs/<run_id>/steps/step_*.json` (LLM-agent trajectory)

Contracts:
- `semantic_opinion_dynamics.contracts.network.NetworkSpec`
- `semantic_opinion_dynamics.contracts.persona.PersonasSpec`
- `semantic_opinion_dynamics.contracts.runlog.StepLog`

## Baseline: DeGroot (suggested approach)

1. Build a row-stochastic influence matrix `W` from `network_used.json`
   - use edge weights; normalize incoming weights per target (or choose a consistent convention)
2. Map each agent’s initial opinion to a scalar `x_i(0)`
   - simple mapping example:
     - `anti_rto=-1`, `neutral=0`, `hybrid=0.2`, `pro_rto=+1`
   - optionally incorporate `confidence` as magnitude or inertia
3. Iterate: `x(t+1) = W x(t)` for the same number of steps as the engine run
4. Compare:
   - baseline scalar trajectory vs. LLM stance/tone trajectory
   - baseline convergence vs. semantic polarization from Researcher B

## Where to put your code

- Put baseline code in this package (e.g., `src/researcher_c/degroot.py`)
- Optional: add small runnable scripts or notebooks, but keep outputs next to the run:
  - `runs/<run_id>/baseline/degroot_series.csv`
  - `runs/<run_id>/baseline/figures/*.png`

## Deliverables

- A reproducible baseline implementation
- A clear experimental story:
  - what changed when you add a leadership bot?
  - what topologies converge vs. polarize?
  - why does the LLM model differ from DeGroot (mechanistically)?
