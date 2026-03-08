#!/usr/bin/env python3
"""Generate advanced visualizations for semantic opinion dynamics snapshots."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NETWORK_JSON = PROJECT_ROOT / "networks" / "Net1_ER.json"
DEFAULT_SLICES_DIR = PROJECT_ROOT / "networks" / "Net1_agent_slices"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "visualization"
REMOTE_RTO_SCALE_LABEL = "Remote-vs-RTO position (0 = Remote, 1 = RTO)"
REMOTE_RTO_TICKS = [0.0, 0.5, 1.0]
REMOTE_RTO_TICK_LABELS = ["Remote", "Hybrid", "RTO"]

TWO_POLE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "two_pole",
    ["#b32134", "#f7f7f7", "#235b9e"],
)

HEATMAP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "heatmap_contrast",
    [
        (0.00, "#7f001b"),
        (0.18, "#b2182b"),
        (0.40, "#ef8a8c"),
        (0.48, "#fffdfd"),
        (0.52, "#fffdfd"),
        (0.60, "#a5c6ec"),
        (0.82, "#2166ac"),
        (1.00, "#08306b"),
    ],
)


@dataclass
class SeriesData:
    steps: List[int]
    node_ids: List[str]
    opinion_matrix: np.ndarray
    graph: nx.Graph
    pagerank: Dict[str, float]
    network_name: str


class SharpMidpointNormalize(mcolors.Normalize):
    """Keep a crisp neutral band around the midpoint and saturate the extremes faster."""

    def __init__(
        self,
        vmin: float = 0.0,
        vmax: float = 1.0,
        midpoint: float = 0.5,
        center_band: float = 0.08,
        exponent: float = 0.7,
        clip: bool = False,
    ) -> None:
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        self.midpoint = midpoint
        self.center_band = center_band
        self.exponent = exponent

    def __call__(self, value, clip=None):  # type: ignore[override]
        values = np.ma.asarray(value, dtype=float)
        clipped = np.clip(values, self.vmin, self.vmax)
        result = np.ma.empty(clipped.shape, dtype=float)

        low_edge = self.midpoint - self.center_band / 2.0
        high_edge = self.midpoint + self.center_band / 2.0

        lower_mask = clipped <= low_edge
        upper_mask = clipped >= high_edge
        center_mask = ~(lower_mask | upper_mask)

        if np.any(lower_mask):
            distance = (low_edge - clipped[lower_mask]) / max(low_edge - self.vmin, 1e-9)
            result[lower_mask] = 0.5 - 0.5 * np.power(distance, self.exponent)
        if np.any(upper_mask):
            distance = (clipped[upper_mask] - high_edge) / max(self.vmax - high_edge, 1e-9)
            result[upper_mask] = 0.5 + 0.5 * np.power(distance, self.exponent)
        if np.any(center_mask):
            center_distance = (clipped[center_mask] - low_edge) / max(high_edge - low_edge, 1e-9)
            result[center_mask] = 0.48 + 0.04 * center_distance

        return np.ma.array(result, mask=np.ma.getmask(values))

    def inverse(self, value):
        values = np.asarray(value, dtype=float)
        result = np.empty_like(values, dtype=float)

        low_edge = self.midpoint - self.center_band / 2.0
        high_edge = self.midpoint + self.center_band / 2.0

        lower_mask = values <= 0.48
        upper_mask = values >= 0.52
        center_mask = ~(lower_mask | upper_mask)

        if np.any(lower_mask):
            distance = np.power(np.clip((0.5 - values[lower_mask]) / 0.5, 0.0, 1.0), 1.0 / self.exponent)
            result[lower_mask] = low_edge - distance * (low_edge - self.vmin)
        if np.any(upper_mask):
            distance = np.power(np.clip((values[upper_mask] - 0.5) / 0.5, 0.0, 1.0), 1.0 / self.exponent)
            result[upper_mask] = high_edge + distance * (self.vmax - high_edge)
        if np.any(center_mask):
            center_distance = np.clip((values[center_mask] - 0.48) / 0.04, 0.0, 1.0)
            result[center_mask] = low_edge + center_distance * (high_edge - low_edge)

        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate advanced figures for opinion-dynamics network snapshots.",
    )
    parser.add_argument(
        "--network-json",
        type=Path,
        default=DEFAULT_NETWORK_JSON,
        help="Base network JSON file.",
    )
    parser.add_argument(
        "--slices-dir",
        type=Path,
        default=DEFAULT_SLICES_DIR,
        help="Directory containing iter*.json snapshot files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figures will be written.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Filename prefix for generated figures. Defaults to the network JSON stem.",
    )
    parser.add_argument(
        "--moving-average-window",
        type=int,
        default=5,
        help="Window size for smoothing the 10th/90th percentile boundaries.",
    )
    parser.add_argument(
        "--camp-threshold",
        type=float,
        default=0.5,
        help="Threshold used to split Remote-leaning (< threshold) and RTO-leaning (>= threshold).",
    )
    parser.add_argument(
        "--max-y-ticks",
        type=int,
        default=30,
        help="Maximum number of Y-axis tick labels for the sorted heatmap.",
    )
    parser.add_argument(
        "--layout-seed",
        type=int,
        default=42,
        help="Random seed used for the topology layout.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--image-format",
        default="png",
        choices=("png", "jpg", "jpeg", "pdf", "svg"),
        help="Output image format.",
    )
    parser.add_argument(
        "--edge-rolling-window",
        type=int,
        default=5,
        help="Rolling-mean window for the influential-edge trend curve.",
    )
    return parser.parse_args()


def natural_key(value: str) -> Tuple[object, ...]:
    parts = re.split(r"(\d+)", str(value))
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def sorted_json_files(directory: Path) -> List[Path]:
    return sorted((path for path in directory.iterdir() if path.suffix == ".json"), key=lambda path: natural_key(path.name))


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_graph(network_data: dict) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(str(node["id"]) for node in network_data.get("nodes", []))

    edge_buckets: Dict[Tuple[str, str], List[float]] = {}
    for node in network_data.get("nodes", []):
        source = str(node["id"])
        neighbors = node.get("neighbors", {})
        for target, weight in neighbors.items():
            target_id = str(target)
            if source == target_id:
                continue
            edge = tuple(sorted((source, target_id), key=natural_key))
            edge_buckets.setdefault(edge, []).append(float(weight))

    for (source, target), weights in edge_buckets.items():
        graph.add_edge(source, target, weight=float(np.mean(weights)))

    return graph


def load_series(network_json: Path, slices_dir: Path) -> SeriesData:
    network_data = load_json(network_json)
    graph = build_graph(network_data)
    node_ids = sorted((str(node["id"]) for node in network_data.get("nodes", [])), key=natural_key)
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}

    files = sorted_json_files(slices_dir)
    if not files:
        raise FileNotFoundError(f"No snapshot JSON files were found in {slices_dir}")

    steps: List[int] = []
    opinion_matrix = np.full((len(node_ids), len(files)), np.nan, dtype=float)

    for column, path in enumerate(files):
        snapshot = load_json(path)
        match = re.search(r"(\d+)", path.stem)
        steps.append(int(match.group(1)) if match else column)
        for node in snapshot.get("nodes", []):
            node_id = str(node["id"])
            if node_id not in node_index:
                continue
            opinion_matrix[node_index[node_id], column] = float(node["opinionScore"])

    if np.isnan(opinion_matrix).any():
        missing_nodes, missing_steps = np.where(np.isnan(opinion_matrix))
        examples = [f"{node_ids[row]}@{steps[col]}" for row, col in zip(missing_nodes[:5], missing_steps[:5])]
        raise ValueError(f"Missing opinionScore values for nodes/steps: {', '.join(examples)}")

    pagerank = nx.pagerank(graph, weight="weight")
    return SeriesData(
        steps=steps,
        node_ids=node_ids,
        opinion_matrix=opinion_matrix,
        graph=graph,
        pagerank=pagerank,
        network_name=network_json.stem,
    )


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    window = min(window, len(values))
    if window <= 1:
        return values.copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def select_tick_positions(values: Sequence[object], max_ticks: int) -> Tuple[np.ndarray, List[str]]:
    if not values:
        return np.array([]), []
    count = min(max_ticks, len(values))
    positions = np.linspace(0, len(values) - 1, count, dtype=int)
    positions = np.unique(positions)
    return positions, [str(values[position]) for position in positions]


def scale_sizes(scores: Dict[str, float], minimum: float = 280.0, maximum: float = 1150.0) -> Dict[str, float]:
    values = np.array([scores[node_id] for node_id in scores], dtype=float)
    if np.allclose(values.max(), values.min()):
        return {node_id: (minimum + maximum) / 2.0 for node_id in scores}
    normalized = (values - values.min()) / (values.max() - values.min())
    scaled = minimum + normalized * (maximum - minimum)
    return {node_id: float(size) for node_id, size in zip(scores, scaled)}


def save_figure(fig: plt.Figure, output_path: Path, dpi: int) -> None:
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def apply_remote_rto_axis(ax: plt.Axes) -> None:
    ax.set_yticks(REMOTE_RTO_TICKS)
    ax.set_yticklabels(REMOTE_RTO_TICK_LABELS)


def apply_remote_rto_colorbar(colorbar) -> None:
    colorbar.set_ticks(REMOTE_RTO_TICKS)
    colorbar.set_ticklabels(REMOTE_RTO_TICK_LABELS)
    colorbar.set_label(REMOTE_RTO_SCALE_LABEL)


def plot_sorted_heatmap(series: SeriesData, output_path: Path, max_y_ticks: int, dpi: int) -> None:
    final_order = np.argsort(series.opinion_matrix[:, -1])
    sorted_matrix = series.opinion_matrix[final_order]
    sorted_ids = [series.node_ids[index] for index in final_order]
    heatmap_norm = SharpMidpointNormalize(vmin=0.0, vmax=1.0, midpoint=0.5, center_band=0.09, exponent=0.72)

    fig, ax = plt.subplots(figsize=(11, 8))
    image = ax.imshow(
        sorted_matrix,
        aspect="auto",
        cmap=HEATMAP_CMAP,
        norm=heatmap_norm,
        origin="lower",
        interpolation="nearest",
    )

    y_positions, y_labels = select_tick_positions(sorted_ids, max_y_ticks)
    x_positions, x_labels = select_tick_positions(series.steps, min(10, len(series.steps)))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Node ID (all nodes shown, ordered by final Remote-vs-RTO position)")
    ax.set_title("Sorted Heatmap of Remote-vs-RTO Trajectories")

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    apply_remote_rto_colorbar(colorbar)
    save_figure(fig, output_path, dpi)


def plot_highlighted_trajectories(series: SeriesData, output_path: Path, dpi: int) -> None:
    steps = np.array(series.steps)
    final_scores = series.opinion_matrix[:, -1]
    min_node = series.node_ids[int(np.argmin(final_scores))]
    max_node = series.node_ids[int(np.argmax(final_scores))]
    influential_nodes = [node_id for node_id, _ in sorted(series.pagerank.items(), key=lambda item: item[1], reverse=True)[:3]]

    highlight_specs: Dict[str, Dict[str, object]] = {
        min_node: {"color": "#c62828", "label": f"Most Remote-leaning at final step: node {min_node}", "linestyle": "-", "linewidth": 2.8},
        max_node: {"color": "#1565c0", "label": f"Most RTO-leaning at final step: node {max_node}", "linestyle": "-", "linewidth": 2.8},
    }
    green_styles = ["-", "--", ":"]
    for rank, node_id in enumerate(influential_nodes, start=1):
        if node_id in highlight_specs:
            highlight_specs[node_id]["label"] = f"{highlight_specs[node_id]['label']} + PageRank #{rank}"
            highlight_specs[node_id]["linewidth"] = 3.0
            continue
        highlight_specs[node_id] = {
            "color": "#2e7d32",
            "label": f"Influential node #{rank}: node {node_id}",
            "linestyle": green_styles[(rank - 1) % len(green_styles)],
            "linewidth": 2.6,
        }

    fig, ax = plt.subplots(figsize=(11.5, 7))
    for row, node_id in enumerate(series.node_ids):
        values = series.opinion_matrix[row]
        if node_id in highlight_specs:
            continue
        ax.plot(steps, values, color="#b3b3b3", linewidth=1.0, alpha=0.45, zorder=1)

    for node_id, spec in highlight_specs.items():
        row = series.node_ids.index(node_id)
        ax.plot(
            steps,
            series.opinion_matrix[row],
            color=spec["color"],
            linestyle=spec["linestyle"],
            linewidth=spec["linewidth"],
            label=spec["label"],
            zorder=3,
        )

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Time step")
    ax.set_ylabel(REMOTE_RTO_SCALE_LABEL)
    apply_remote_rto_axis(ax)
    ax.set_title("Highlighted Remote-vs-RTO Trajectories")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, loc="best")
    save_figure(fig, output_path, dpi)


def plot_shaded_median_area(series: SeriesData, output_path: Path, window: int, dpi: int) -> None:
    steps = np.array(series.steps)
    lower_band = np.quantile(series.opinion_matrix, 0.10, axis=0)
    upper_band = np.quantile(series.opinion_matrix, 0.90, axis=0)
    median_line = np.median(series.opinion_matrix, axis=0)

    smoothed_lower = moving_average(lower_band, window)
    smoothed_upper = moving_average(upper_band, window)

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    ax.fill_between(
        steps,
        smoothed_lower,
        smoothed_upper,
        color="#9ecae1",
        alpha=0.45,
        label="10th-90th percentile band",
    )
    ax.plot(steps, median_line, color="#1f1f1f", linewidth=2.6, label="Median opinion")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Opinion score")
    ax.set_title("Median Trajectory with Smoothed Dispersion Band")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, loc="best")
    save_figure(fig, output_path, dpi)


def edge_influence_series(series: SeriesData) -> Dict[Tuple[str, str], np.ndarray]:
    node_index = {node_id: idx for idx, node_id in enumerate(series.node_ids)}
    influences: Dict[Tuple[str, str], np.ndarray] = {}
    for source, target, data in series.graph.edges(data=True):
        weight = float(data.get("weight", 1.0))
        values = weight * np.abs(
            series.opinion_matrix[node_index[source]] - series.opinion_matrix[node_index[target]]
        )
        influences[(source, target)] = values
    return influences


def select_most_influential_edge(influences: Dict[Tuple[str, str], np.ndarray]) -> Tuple[Tuple[str, str], np.ndarray]:
    return max(
        influences.items(),
        key=lambda item: (float(np.sum(item[1])), float(np.max(item[1])), natural_key(item[0][0]), natural_key(item[0][1])),
    )


def plot_most_influential_edge(
    series: SeriesData,
    output_path: Path,
    rolling_window: int,
    dpi: int,
) -> Tuple[Tuple[str, str], np.ndarray]:
    influences = edge_influence_series(series)
    edge, values = select_most_influential_edge(influences)
    weight = float(series.graph.edges[edge]["weight"])
    steps = np.array(series.steps)
    smoothed_values = moving_average(values, rolling_window)

    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    ax.plot(steps, values, color="#d97706", linewidth=1.2, alpha=0.4, label="Raw influence")
    ax.plot(
        steps,
        smoothed_values,
        color="#92400e",
        linewidth=2.9,
        label=f"Rolling mean (window={min(max(rolling_window, 1), len(values))})",
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel(r"$C_{u \leftrightarrow v}(t) = w_{uv} \cdot |x_u(t) - x_v(t)|$")
    ax.set_title(f"Influence Over Time for the Most Influential Edge ({edge[0]}-{edge[1]})")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="best")
    ax.text(
        0.02,
        0.98,
        f"weight = {weight:.3f}\ncumulative influence = {np.sum(values):.3f}\npeak influence = {np.max(values):.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#d9d9d9", "boxstyle": "round,pad=0.35"},
    )
    save_figure(fig, output_path, dpi)
    return edge, values


def fallback_edge_correlation(graph: nx.Graph, opinions: Dict[str, float]) -> float:
    source_values = np.array([opinions[source] for source, _ in graph.edges()], dtype=float)
    target_values = np.array([opinions[target] for _, target in graph.edges()], dtype=float)
    if source_values.size == 0:
        return float("nan")
    if np.allclose(source_values, source_values[0]) and np.allclose(target_values, target_values[0]):
        return 1.0 if math.isclose(float(source_values[0]), float(target_values[0])) else 0.0
    if np.std(source_values) == 0.0 or np.std(target_values) == 0.0:
        return 0.0
    return float(np.corrcoef(source_values, target_values)[0, 1])


def compute_echo_chamber_index(series: SeriesData) -> np.ndarray:
    values = []
    node_index = {node_id: idx for idx, node_id in enumerate(series.node_ids)}
    for column in range(series.opinion_matrix.shape[1]):
        opinions = {node_id: float(series.opinion_matrix[node_index[node_id], column]) for node_id in series.node_ids}
        nx.set_node_attributes(series.graph, opinions, "opinion")
        coefficient = nx.numeric_assortativity_coefficient(series.graph, "opinion")
        if np.isnan(coefficient):
            coefficient = fallback_edge_correlation(series.graph, opinions)
        values.append(float(coefficient))
    return np.array(values, dtype=float)


def plot_echo_chamber_index(series: SeriesData, output_path: Path, dpi: int) -> np.ndarray:
    steps = np.array(series.steps)
    values = compute_echo_chamber_index(series)

    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    ax.plot(steps, values, color="#37474f", linewidth=2.5)
    ax.axhline(0.0, color="#9e9e9e", linestyle="--", linewidth=1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Pearson correlation on edge endpoints")
    ax.set_title("Echo-Chamber Index Over Time")
    ax.grid(alpha=0.18)
    save_figure(fig, output_path, dpi)
    return values


def compute_cross_cutting_ratio(series: SeriesData, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    node_index = {node_id: idx for idx, node_id in enumerate(series.node_ids)}
    total_edges = max(series.graph.number_of_edges(), 1)
    cross = []
    internal = []

    for column in range(series.opinion_matrix.shape[1]):
        opinions = {node_id: float(series.opinion_matrix[node_index[node_id], column]) for node_id in series.node_ids}
        cross_edges = 0
        for source, target in series.graph.edges():
            source_camp = opinions[source] >= threshold
            target_camp = opinions[target] >= threshold
            if source_camp != target_camp:
                cross_edges += 1
        cross_ratio = 100.0 * cross_edges / total_edges
        cross.append(cross_ratio)
        internal.append(100.0 - cross_ratio)

    return np.array(internal, dtype=float), np.array(cross, dtype=float)


def plot_cross_cutting_ratio(series: SeriesData, output_path: Path, threshold: float, dpi: int) -> np.ndarray:
    internal, cross = compute_cross_cutting_ratio(series, threshold)
    steps = np.array(series.steps)

    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    ax.stackplot(
        steps,
        internal,
        cross,
        labels=["Internal connections", "Cross-cutting connections"],
        colors=["#cbd5e1", "#f59e0b"],
        alpha=0.95,
    )
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Edge ratio (%)")
    ax.set_title(f"Internal vs Cross-Cutting Edge Ratio (threshold = {threshold:.2f})")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.18)
    save_figure(fig, output_path, dpi)
    return cross


def compute_bounds(position_sets: Iterable[Dict[str, Tuple[float, float]]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xs: List[float] = []
    ys: List[float] = []
    for positions in position_sets:
        for x_coord, y_coord in positions.values():
            xs.append(x_coord)
            ys.append(y_coord)
    x_pad = (max(xs) - min(xs)) * 0.14 + 0.25
    y_pad = (max(ys) - min(ys)) * 0.18 + 0.25
    return (min(xs) - x_pad, max(xs) + x_pad), (min(ys) - y_pad, max(ys) + y_pad)


def add_cluster_cloud(
    ax: plt.Axes,
    positions: Dict[str, Tuple[float, float]],
    nodes: List[str],
    color: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> None:
    if not nodes:
        return
    x_grid = np.linspace(xlim[0], xlim[1], 180)
    y_grid = np.linspace(ylim[0], ylim[1], 180)
    xx, yy = np.meshgrid(x_grid, y_grid)
    density = np.zeros_like(xx)

    sigma_x = max((xlim[1] - xlim[0]) * 0.055, 0.11)
    sigma_y = max((ylim[1] - ylim[0]) * 0.055, 0.11)
    for node_id in nodes:
        x_coord, y_coord = positions[node_id]
        density += np.exp(
            -(
                ((xx - x_coord) ** 2) / (2.0 * sigma_x ** 2)
                + ((yy - y_coord) ** 2) / (2.0 * sigma_y ** 2)
            )
        )

    peak = float(density.max())
    if peak <= 0.0:
        return
    density /= peak

    levels = [0.08, 0.18, 0.32, 0.50, 0.72, 1.01]
    contour_colors = [
        mcolors.to_rgba(color, alpha)
        for alpha in (0.18, 0.24, 0.30, 0.38, 0.48)
    ]
    ax.contourf(xx, yy, density, levels=levels, colors=contour_colors, antialiased=True, zorder=0)
    ax.contour(xx, yy, density, levels=[0.18, 0.50, 0.72], colors=[color], linewidths=1.0, alpha=0.55, zorder=0)


def draw_network_panel(
    ax: plt.Axes,
    graph: nx.Graph,
    positions: Dict[str, Tuple[float, float]],
    opinions: Dict[str, float],
    sizes: Dict[str, float],
    threshold: float,
    title: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> None:
    camp_a = [node_id for node_id, value in opinions.items() if value < threshold]
    camp_b = [node_id for node_id, value in opinions.items() if value >= threshold]

    add_cluster_cloud(ax, positions, camp_a, "#f4bcc2", xlim, ylim)
    add_cluster_cloud(ax, positions, camp_b, "#bfd7fb", xlim, ylim)

    for source, target in graph.edges():
        xs = [positions[source][0], positions[target][0]]
        ys = [positions[source][1], positions[target][1]]
        ax.plot(xs, ys, color="#9e9e9e", linewidth=0.85, alpha=0.42, zorder=1)

    node_ids = sorted(graph.nodes(), key=natural_key)
    x_points = [positions[node_id][0] for node_id in node_ids]
    y_points = [positions[node_id][1] for node_id in node_ids]
    color_values = [opinions[node_id] for node_id in node_ids]
    node_sizes = [sizes[node_id] for node_id in node_ids]
    ax.scatter(
        x_points,
        y_points,
        c=color_values,
        s=node_sizes,
        cmap=TWO_POLE_CMAP,
        vmin=0.0,
        vmax=1.0,
        edgecolors="white",
        linewidths=0.9,
        alpha=0.97,
        zorder=2,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.axis("off")
    ax.text(
        0.02,
        0.02,
        f"Remote-leaning: {len(camp_a)}  |  RTO-leaning: {len(camp_b)}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#4a4a4a",
    )


def plot_distribution_infographic(series: SeriesData, output_path: Path, threshold: float, layout_seed: int, dpi: int) -> None:
    base_positions = {
        node_id: (float(coords[0]), float(coords[1]))
        for node_id, coords in nx.spring_layout(series.graph, seed=layout_seed, weight="weight").items()
    }
    node_index = {node_id: idx for idx, node_id in enumerate(series.node_ids)}
    start_opinions = {node_id: float(series.opinion_matrix[node_index[node_id], 0]) for node_id in series.node_ids}
    end_opinions = {node_id: float(series.opinion_matrix[node_index[node_id], -1]) for node_id in series.node_ids}

    xlim, ylim = compute_bounds([base_positions])
    sizes = scale_sizes(series.pagerank)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.6), facecolor="white")
    fig.suptitle("Remote vs. RTO Network Distribution: Beginning vs End", fontsize=16, y=0.98)

    draw_network_panel(
        axes[0],
        series.graph,
        base_positions,
        start_opinions,
        sizes,
        threshold,
        f"Beginning (step {series.steps[0]})",
        xlim,
        ylim,
    )
    draw_network_panel(
        axes[1],
        series.graph,
        base_positions,
        end_opinions,
        sizes,
        threshold,
        f"End (step {series.steps[-1]})",
        xlim,
        ylim,
    )

    colorbar = fig.colorbar(
        ScalarMappable(norm=mcolors.Normalize(vmin=0.0, vmax=1.0), cmap=TWO_POLE_CMAP),
        ax=axes,
        fraction=0.036,
        pad=0.02,
    )
    apply_remote_rto_colorbar(colorbar)
    save_figure(fig, output_path, dpi)


def generate_all_figures(args: argparse.Namespace) -> List[Path]:
    network_json = args.network_json.resolve()
    slices_dir = args.slices_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    series = load_series(network_json, slices_dir)
    prefix = args.prefix or series.network_name
    extension = args.image_format

    outputs = {
        "heatmap": output_dir / f"{prefix}_Sorted_Heatmap.{extension}",
        "trajectories": output_dir / f"{prefix}_Highlighted_Trajectories.{extension}",
        "median_band": output_dir / f"{prefix}_Shaded_Median_Area.{extension}",
        "edge_influence": output_dir / f"{prefix}_Most_Influential_Edge.{extension}",
        "distribution": output_dir / f"{prefix}_Opinion_Distribution_Beginning_End.{extension}",
        "echo_chamber": output_dir / f"{prefix}_Echo_Chamber_Index.{extension}",
        "cross_cutting": output_dir / f"{prefix}_Cross_Cutting_Edge_Ratio.{extension}",
    }

    plot_sorted_heatmap(series, outputs["heatmap"], args.max_y_ticks, args.dpi)
    plot_highlighted_trajectories(series, outputs["trajectories"], args.dpi)
    plot_shaded_median_area(series, outputs["median_band"], args.moving_average_window, args.dpi)
    edge, _ = plot_most_influential_edge(series, outputs["edge_influence"], args.edge_rolling_window, args.dpi)
    plot_distribution_infographic(series, outputs["distribution"], args.camp_threshold, args.layout_seed, args.dpi)
    echo_values = plot_echo_chamber_index(series, outputs["echo_chamber"], args.dpi)
    cross_values = plot_cross_cutting_ratio(series, outputs["cross_cutting"], args.camp_threshold, args.dpi)

    print(f"Loaded {series.network_name} with {len(series.node_ids)} nodes and {len(series.steps)} snapshots.")
    print(f"Most influential edge by cumulative influence: {edge[0]}-{edge[1]}")
    print(
        "Top 3 influential nodes by weighted PageRank: "
        + ", ".join(node_id for node_id, _ in sorted(series.pagerank.items(), key=lambda item: item[1], reverse=True)[:3])
    )
    print(f"Final echo-chamber index: {echo_values[-1]:.4f}")
    print(f"Final cross-cutting edge ratio: {cross_values[-1]:.2f}%")
    for output_path in outputs.values():
        print(f"Saved {output_path}")

    return list(outputs.values())


def main() -> None:
    args = parse_args()
    generate_all_figures(args)


if __name__ == "__main__":
    main()
