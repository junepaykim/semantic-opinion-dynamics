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
- `OpenAI`: put your API key in `api_key.txt` at the project root, or set `OPENAI_API_KEY`.
- `Ollama`: local fallback backend. Default model in code is `qwen3:4b`.

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
│   ├── Net1_ER.json
│   ├── Net2_SW.json
│   ├── Net3_SF.json
│   ├── Net4_KC.json
│   └── {name}_{model}_slices/
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

Each iteration is saved under `networks/{name}_{model}_slices/`:

```text
networks/myNet_agent_slices/
├── iter0.json
├── iter1.json
├── iter2.json
└── ...
```

## 4. Workflow

### 4.1 Generate a network

`main.py` can generate one of four graph types:
- `random`
- `small_world`
- `scale_free`
- `karate_club`

Generation flow:
- build graph topology
- initialize node `opinionScore`
- generate `persona`
- generate opinion `prompt`
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
| `--model` | `agent` or `degroot` | `agent` |
| `--iters` | Number of iterations | required when running |

### 5.2 Examples

```bash
# Generate a 20-node random network
python main.py -g -t random -n myNet

# Generate a 50-node small-world network
python main.py -g -t small_world -n mySW --nodes 50

# Run 10 agent iterations
python main.py -n Net1_ER --model agent --iters 10

# Run 20 DeGroot iterations
python main.py -n Net1_ER --model degroot --iters 20
```

## 6. Advanced Visualization

The recommended visualization entry point is [advanced_network_visualizations.py](src/visualization/advanced_network_visualizations.py).

It loads:
- a base network JSON
- a snapshot directory such as `networks/Net1_agent_slices`

It then generates the following figures:

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

### 6.3 Distribution View

- `Opinion Distribution: Beginning vs End`
  - same node layout in both panels
  - node colors map from Remote to RTO
  - pastel cluster shading for Remote-leaning and RTO-leaning groups

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
| `--network-json` | Base network JSON file | `networks/Net1_ER.json` |
| `--slices-dir` | Snapshot directory | `networks/Net1_agent_slices` |
| `--output-dir` | Figure output directory | `src/visualization` |
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
# Use Net1 defaults
python src/visualization/advanced_network_visualizations.py

# Generate figures for another snapshot directory
python src/visualization/advanced_network_visualizations.py \
  --network-json networks/Net2_SW.json \
  --slices-dir networks/Net2_agent_slices \
  --prefix Net2_SW

# Save as SVG with a 3-step rolling mean for edge influence
python src/visualization/advanced_network_visualizations.py \
  --image-format svg \
  --edge-rolling-window 3
```

### 6.7 Expected Outputs

For `Net1_ER`, the script writes files such as:

```text
src/visualization/Net1_ER_Sorted_Heatmap.png
src/visualization/Net1_ER_Highlighted_Trajectories.png
src/visualization/Net1_ER_Shaded_Median_Area.png
src/visualization/Net1_ER_Most_Influential_Edge.png
src/visualization/Net1_ER_Opinion_Distribution_Beginning_End.png
src/visualization/Net1_ER_Echo_Chamber_Index.png
src/visualization/Net1_ER_Cross_Cutting_Edge_Ratio.png
```

## 7. Module Notes

- [main.py](main.py)
  - CLI entry point for generation and iteration

- [networkOps.py](src/input/networkOps.py)
  - graph generation, load/save, initial node setup

- [modelCall.py](src/input/modelCall.py)
  - LLM prompts and score-to-text generation
  - opinion scale is explicitly defined as `0 = Remote`, `1 = Office/RTO`

- [src/model/agentModel/iterate.py](src/model/agentModel/iterate.py)
  - LLM-based iterative update

- [src/model/baseline/iterate.py](src/model/baseline/iterate.py)
  - DeGroot baseline update

- [advanced_network_visualizations.py](src/visualization/advanced_network_visualizations.py)
  - current advanced visualization pipeline

## 8. Current Status

Implemented in the repository now:
- network generation and persistence
- LLM-based agent iteration
- DeGroot baseline iteration
- saved per-step network slices
- advanced visualization for trajectory, influence, distribution, and polarization analysis

The older [run_all_analysis.py](src/visualization/run_all_analysis.py) remains in the repository as an earlier analysis script, but `advanced_network_visualizations.py` is the current script to use for the new figure set.
