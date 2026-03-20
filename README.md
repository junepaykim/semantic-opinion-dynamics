# Semantic Opinion Dynamics

## 0. Installation

**Requirements:** Python 3.10+

```bash
# Create virtual environment (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Current Python dependencies in [requirements.txt](requirements.txt):
- `networkx`
- `numpy`
- `matplotlib`
- `scipy`
- `requests`
- `tqdm`

**LLM backends**
- `OpenAI`: put your API key in `api_key.txt` at the project root, or set `OPENAI_API_KEY`. Uses a model fallback chain (`gpt-3.5-turbo` → `gpt-4o-mini` → `gpt-4o`) when context or rate limits are hit.
- `Ollama`: local fallback backend when OpenAI fails. Default model in code is `qwen3:4b`.

## 1. Overview

This project studies **opinion dynamics on social networks** using the domain:

- `0.0` = strongly prefer **Remote Work**
- `1.0` = strongly prefer **Return to Office (RTO)**

Each node is an LLM agent with:
- a continuous `opinionScore`
- a short natural-language opinion `prompt`
- a `persona`
- weighted neighbor connections

The repository supports:
- network generation
- iterative opinion updates with an LLM-based agent model
- iterative updates with a DeGroot baseline
- advanced visualization of node trajectories, influence, distribution, and polarization

## 2. Project Structure

```text
semantic-opinion-dynamics/
├── main.py
├── requirements.txt
├── networks/
│   ├── Net_*.json                    # e.g. Net_random_skew_right_1_ER_SR1.json
│   └── Net_*_{agent|degroot}_slices/
├── plots/                            # Legacy: Net1_ER, Net2_SW, Net3_SF, Net4_KC (normal dist)
│   ├── Net1_ER.json, Net2_SW.json, ...
│   ├── Net1_agent_slices, Net1_ER_degroot_slices, ...
├── src/
│   ├── input/
│   │   ├── __init__.py
│   │   ├── modelCall.py
│   │   └── networkOps.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── agentModel/
│   │   │   ├── __init__.py
│   │   │   └── iterate.py
│   │   └── baseline/
│   │       ├── __init__.py
│   │       └── iterate.py
│   └── visualization/
│       ├── advanced_network_visualizations.py
│       ├── output/                   # Generated figures: {ER|SW|SF|KC}/{N|SR1|SR2|SR3|P}/{agent|degroot}/
│       ├── run_all_analysis.py
│       └── *.png / *.jpg
└── assets/
```

## 3. Data Format

### 3.1 Network JSON

```json
{
  "nodes": [
    {
      "id": "1",
      "opinionScore": 0.2,
      "prompt": "I prefer remote work because ...",
      "persona": "gentle, patient, balanced",
      "neighbors": {
        "2": 0.6,
        "3": 0.4
      }
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Node identifier |
| `opinionScore` | `float` in `[0,1]` | `0 = Remote`, `1 = RTO` |
| `prompt` | `str` | Short first-person opinion text |
| `persona` | `str` | Comma-separated persona descriptors |
| `neighbors` | `dict[str, float]` | Neighbor ID to edge-weight mapping |

### 3.2 Snapshot Slices

Each iteration is saved under `networks/{name}_{model}_slices/` or `plots/`:

```text
networks/Net_random_skew_right_1_ER_SR1_agent_slices/
├── iter0.json
├── iter1.json
├── iter2.json
└── ...
```

**Naming convention:** `{base}_{graph_type}_{score_dist}` e.g. `Net_random_skew_right_1_ER_SR1` (ER=random, SR1=skew_right_1).

## 4. Workflow

### 4.1 Generate a network

`main.py` can generate one of four graph types:
- `random` (ER)
- `small_world` (SW)
- `scale_free` (SF)
- `karate_club` (KC)

**Opinion score distributions** (`--score-dist`):
- `normal` (N)
- `skew_left_1`, `skew_left_2`, `skew_left_3` (SL1–SL3)
- `skew_right_1`, `skew_right_2`, `skew_right_3` (SR1–SR3)
- `polarized` (P): bimodal, half near 0 and half near 1

Generation flow:
- build graph topology
- sample node `opinionScore` from the chosen distribution
- generate `persona` via LLM
- generate opinion `prompt` via LLM
- save initial network JSON into `networks/`

### 4.2 Run an opinion model

Supported models:
- `agent`: LLM-based update using persona and neighbor prompts
- `degroot`: weighted averaging baseline

Runtime flow:
- load base network JSON
- save `iter0.json`
- run `N` iterations
- save each step into `networks/{name}_{model}_slices/`

## 5. Main CLI

### 5.1 Arguments

| Argument | Description | Default |
|---|---|---|
| `-g`, `--generate` | Generate a new network | `False` |
| `-t`, `--graph-type` | `random`, `small_world`, `scale_free`, `karate_club` | `random` |
| `-n`, `--name` | Base network name | required when running |
| `--nodes` | Number of nodes when generating | `20` |
| `--score-dist` | Opinion distribution: `normal`, `skew_left_1/2/3`, `skew_right_1/2/3`, `polarized` | `normal` |
| `--model` | `agent` or `degroot` | `agent` |
| `--iters` | Number of iterations | required when running |

### 5.2 Examples

```bash
# Generate a 20-node random network (normal distribution)
python main.py -g -t random -n myNet

# Generate a 50-node small-world network with polarized opinions
python main.py -g -t small_world -n mySW --nodes 50 --score-dist polarized

# Generate skew_right_1 random network
python main.py -g -t random -n Net_random_skew_right_1 --nodes 50 --score-dist skew_right_1

# Run 50 agent iterations
python main.py -n Net_random_skew_right_1_ER_SR1 --model agent --iters 50

# Run 50 DeGroot iterations
python main.py -n Net_random_skew_right_1_ER_SR1 --model degroot --iters 50
```

## 6. Advanced Visualization

The recommended visualization entry point is [advanced_network_visualizations.py](src/visualization/advanced_network_visualizations.py).

**Data sources:** The script auto-discovers (network JSON, slices dir) pairs from:
- `networks/` (e.g. `Net_random_skew_right_1_ER_SR1.json` + `*_agent_slices` / `*_degroot_slices`)
- `plots/` (legacy: `Net1_ER`, `Net2_SW`, `Net3_SF`, `Net4_KC` with both agent and degroot slices)

**Output structure:** Figures are saved under `src/visualization/output/{graph_type}/{score_dist}/{iteration}/`:
- `graph_type`: ER, SW, SF, KC
- `score_dist`: N, SR1, SR2, SR3, P, etc.
- `iteration`: agent, degroot

**Default behavior:** Running the script with no arguments processes all discovered pairs.

### 6.1 Trajectory Views

- `Sorted Heatmap`
  - all nodes shown
  - ordered by final Remote-vs-RTO position
  - color scale labeled `Remote / Hybrid / RTO`

- `Highlighted Trajectories`
  - most Remote-leaning node
  - most RTO-leaning node
  - top 3 influential nodes
  - remaining trajectories in gray

- `Shaded Median Area`
  - median line
  - shaded band between the 10th and 90th percentiles
  - moving-average smoothing on the band boundaries

### 6.2 Influence View

- `Most Influential Edge`
  - computes edge influence using:
    - `C_{u,v}(t) = w_uv * |x_u(t) - x_v(t)|`
  - selects the edge with maximum cumulative influence over time
  - plots raw influence plus a rolling-mean trend line

### 6.3 Distribution Views

- `Opinion Distribution: Beginning vs End`
  - same node layout in both panels
  - node colors map from Remote to RTO
  - pastel cluster shading for Remote-leaning and RTO-leaning groups

- `Opinion Score Histogram`
  - bar chart of opinion score distribution (bins 0–0.1, 0.1–0.2, …)
  - compares initial vs final state

### 6.4 Polarization Views

- `Echo-Chamber Index`
  - network assortativity / Pearson-style correlation on edge endpoints
  - y-axis range `[-1, 1]`

- `Cross-Cutting Edge Ratio`
  - threshold split at `0.5` by default
  - stacked area chart of:
    - internal same-side connections
    - cross-cutting Remote-vs-RTO connections

### 6.5 Advanced Visualization CLI

| Argument | Description | Default |
|---|---|---|
| `--network-json` | Base network JSON file | auto-discovered |
| `--slices-dir` | Snapshot directory | auto-discovered |
| `--output-dir` | Figure output directory | `src/visualization/output` |
| `--single` | Process only the first discovered pair | `False` |
| `--prefix` | Output filename prefix | network stem |
| `--moving-average-window` | Smoothing for percentile band | `5` |
| `--camp-threshold` | Remote/RTO split threshold | `0.5` |
| `--max-y-ticks` | Max heatmap Y labels | `30` |
| `--layout-seed` | Seed for network layout | `42` |
| `--dpi` | Figure DPI | `220` |
| `--image-format` | `png`, `jpg`, `jpeg`, `pdf`, `svg` | `png` |
| `--edge-rolling-window` | Rolling mean for edge influence | `5` |

### 6.6 Advanced Visualization Examples

```bash
# Process all discovered networks (default)
python src/visualization/advanced_network_visualizations.py

# Process only the first discovered pair
python src/visualization/advanced_network_visualizations.py --single

# Specify a single network explicitly
python src/visualization/advanced_network_visualizations.py \
  --network-json networks/Net_random_skew_right_1_ER_SR1.json \
  --slices-dir networks/Net_random_skew_right_1_ER_SR1_agent_slices

# Save as SVG with a 3-step rolling mean for edge influence
python src/visualization/advanced_network_visualizations.py \
  --image-format svg \
  --edge-rolling-window 3
```

### 6.7 Expected Outputs

For each network, the script writes 8 figures per (network, slices) pair:

```text
src/visualization/output/ER/N/agent/Net1_ER_Sorted_Heatmap.png
src/visualization/output/ER/N/agent/Net1_ER_Highlighted_Trajectories.png
src/visualization/output/ER/N/agent/Net1_ER_Shaded_Median_Area.png
src/visualization/output/ER/N/agent/Net1_ER_Most_Influential_Edge.png
src/visualization/output/ER/N/agent/Net1_ER_Opinion_Distribution_Beginning_End.png
src/visualization/output/ER/N/agent/Net1_ER_Opinion_Score_Histogram.png
src/visualization/output/ER/N/agent/Net1_ER_Echo_Chamber_Index.png
src/visualization/output/ER/N/agent/Net1_ER_Cross_Cutting_Edge_Ratio.png
```

## 7. Module Notes

- [main.py](main.py)
  - CLI entry point for generation and iteration

- [networkOps.py](src/input/networkOps.py)
  - graph generation, load/save, initial node setup
  - opinion score sampling (normal, skew_left, skew_right, polarized)

- [modelCall.py](src/input/modelCall.py)
  - LLM prompts and score-to-text generation
  - opinion scale is explicitly defined as `0 = Remote`, `1 = Office/RTO`
  - OpenAI model fallback chain (gpt-3.5-turbo → gpt-4o-mini → gpt-4o) on context/rate limit errors

- [src/model/agentModel/iterate.py](src/model/agentModel/iterate.py)
  - LLM-based iterative update

- [src/model/baseline/iterate.py](src/model/baseline/iterate.py)
  - DeGroot baseline update

- [advanced_network_visualizations.py](src/visualization/advanced_network_visualizations.py)
  - current advanced visualization pipeline

## 8. Current Status

Implemented in the repository now:
- network generation with multiple opinion score distributions (normal, skew_left, skew_right, polarized)
- LLM-based agent iteration with OpenAI model fallback (context/rate limit handling)
- DeGroot baseline iteration
- saved per-step network slices
- advanced visualization for trajectory, influence, distribution, histogram, and polarization analysis
- auto-discovery from `networks/` and `plots/` with output organized by graph type, score distribution, and iteration model

The older [run_all_analysis.py](src/visualization/run_all_analysis.py) remains in the repository as an earlier analysis script, but `advanced_network_visualizations.py` is the current script to use for the new figure set.
