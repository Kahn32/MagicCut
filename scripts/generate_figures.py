#!/usr/bin/env python3
"""Generate publication-quality figures from immutable locked evaluation results."""

from __future__ import annotations

import json
import tarfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import precision_recall_curve

from magiccut.adaptive import QueryCase
from magiccut.data.renders import parse_render_name
from magiccut.frozen import GRAPH, PROBABILITY_THRESHOLD, VAL_UIDS
from magiccut.graph import build_knn_graph, weighted_edges
from magiccut.io import read_json, sha256_file, write_json


COLORS = {
    "mmw": "#6B7280",
    "adaptive": "#2F7DBD",
    "calibrated": "#8A5CB7",
    "magiccut": "#E0792D",
    "active": "#15956A",
    "target": "#15956A",
    "false_positive": "#D64B4B",
    "false_negative": "#F2C14E",
    "background": "#F7F5EF",
    "ink": "#18212A",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def save(fig, output: Path, name: str) -> list[Path]:
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"{name}.{suffix}"
        fig.savefig(path, dpi=320 if suffix == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def method_diagram(output: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.2, "Part renders", "Three nested crops"),
        (2.15, "Frozen encoder", "1152-D feature"),
        (4.1, "Calibrated unary", "Material likelihood"),
        (6.05, "MagicCut", "kNN Potts graph"),
        (8.0, "Clarification", "Yes / no constraint"),
    ]
    for x, title, subtitle in boxes:
        patch = FancyBboxPatch((x, 0.9), 1.65, 1.15, boxstyle="round,pad=0.08", fc="#F7F5EF", ec="#AAB3BC")
        ax.add_patch(patch)
        ax.text(x + 0.825, 1.6, title, ha="center", va="center", weight="bold", color=COLORS["ink"])
        ax.text(x + 0.825, 1.27, subtitle, ha="center", va="center", fontsize=8, color="#66727D")
    for x in (1.85, 3.8, 5.75, 7.7):
        ax.add_patch(FancyArrowPatch((x, 1.48), (x + 0.28, 1.48), arrowstyle="-|>", mutation_scale=12, color="#66727D"))
    ax.add_patch(FancyArrowPatch((8.85, 0.86), (6.85, 0.86), connectionstyle="arc3,rad=.25", arrowstyle="-|>", mutation_scale=12, color=COLORS["active"]))
    ax.text(7.85, 0.33, "rerun graph cut with a hard constraint", ha="center", fontsize=8, color=COLORS["active"])
    ax.set_title("MagicCut: calibrated graph inference with active binary feedback", loc="left", weight="bold")
    return save(fig, output, "method_diagram")


def candidate_arrays(rows: list[dict], method: str):
    labels, scores = [], []
    for row in rows:
        selected = set()
        if method == "magiccut":
            selected = set(row["methods"]["magiccut"]["selected_ids"])
        elif method == "active":
            selected = set(row["active_history"][-1]["selected_ids"])
        for item in row["candidate_records"]:
            if item["part_id"] == row["query"]:
                continue
            labels.append(int(item["is_target"]))
            if method == "mmw":
                scores.append(-item["distance"])
            elif method == "calibrated":
                scores.append(item["calibrated_probability"])
            else:
                scores.append(int(item["part_id"] in selected) + 1e-6 * item["calibrated_probability"])
    return np.asarray(labels), np.asarray(scores)


def precision_recall(rows: list[dict], output: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    for method, label, color in (
        ("mmw", "Material Magic Wand", COLORS["mmw"]),
        ("calibrated", "Calibrated retrieval", COLORS["calibrated"]),
        ("magiccut", "MagicCut partition", COLORS["magiccut"]),
        ("active", "MagicCut + 3 clicks", COLORS["active"]),
    ):
        y, score = candidate_arrays(rows, method)
        precision, recall, _ = precision_recall_curve(y, score)
        ax.plot(recall, precision, label=label, lw=2, color=color)
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.01), ylim=(0, 1.01), title="Held-out candidate precision–recall")
    ax.grid(alpha=.18)
    ax.legend(frameon=False, fontsize=8)
    return save(fig, output, "precision_recall")


def f1_clicks(report35: dict, output: Path) -> list[Path]:
    curve = report35["heldout_94_meshes"]["active"]
    baseline = report35["heldout_94_meshes"]["methods"]["material_magic_wand"]["f1"]
    magiccut = report35["heldout_94_meshes"]["methods"]["magiccut"]["f1"]
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    clicks = [row["click"] for row in curve]
    ax.plot(clicks, [row["f1"] for row in curve], marker="o", lw=2.2, color=COLORS["active"], label="Influence-aware")
    ax.axhline(baseline, ls="--", color=COLORS["mmw"], label="Material Magic Wand")
    ax.axhline(magiccut, ls=":", color=COLORS["magiccut"], label="MagicCut, no feedback")
    ax.set(xticks=clicks, xlabel="User clicks", ylabel="Mean grouping F1", title="F1 versus clarification budget")
    ax.grid(alpha=.18)
    ax.legend(frameon=False, fontsize=8)
    return save(fig, output, "f1_versus_clicks")


def calibration(rows: list[dict], output: Path) -> list[Path]:
    y, probability = [], []
    for row in rows:
        for item in row["candidate_records"]:
            if item["part_id"] != row["query"]:
                y.append(int(item["is_target"]))
                probability.append(item["calibrated_probability"])
    y, probability = np.asarray(y), np.asarray(probability)
    edges = np.linspace(0, 1, 11)
    points = []
    for index in range(10):
        mask = (probability >= edges[index]) & (probability < edges[index + 1] if index < 9 else probability <= 1)
        if mask.any():
            points.append((probability[mask].mean(), y[mask].mean(), int(mask.sum())))
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.plot([0, 1], [0, 1], "--", color="#9AA4AE", lw=1, label="Perfect calibration")
    ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", lw=2, color=COLORS["calibrated"], label="Held-out bins")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed positive rate", xlim=(0, 1), ylim=(0, 1), title="Reliability diagram")
    ax.grid(alpha=.18)
    ax.legend(frameon=False, fontsize=8)
    return save(fig, output, "calibration")


def risk_coverage(rows: list[dict], output: Path) -> list[Path]:
    labels, probabilities, normalized = [], [], []
    for row in rows:
        for item in row["candidate_records"]:
            if item["part_id"] == row["query"]:
                continue
            labels.append(int(item["is_target"]))
            probabilities.append(item["calibrated_probability"])
            normalized.append(item["normalized_distance"])
    labels = np.asarray(labels)
    probability = np.asarray(probabilities)
    predicted = probability >= PROBABILITY_THRESHOLD
    errors = predicted != labels
    confidence = np.abs(probability - PROBABILITY_THRESHOLD)
    order = np.argsort(-confidence)
    coverage = np.arange(1, len(order) + 1) / len(order)
    risk = np.cumsum(errors[order]) / np.arange(1, len(order) + 1)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.plot(coverage, risk, lw=2.2, color=COLORS["calibrated"], label="Calibration margin")
    ax.axhline(errors.mean(), ls="--", color=COLORS["mmw"], label="No rejection")
    ax.set(xlabel="Coverage", ylabel="Classification risk", xlim=(0, 1), ylim=(0, max(.02, risk.max() * 1.05)), title="Held-out risk–coverage")
    ax.grid(alpha=.18)
    ax.legend(frameon=False, fontsize=8)
    return save(fig, output, "risk_coverage")


def load_embedding_case(row: dict, manifest: dict) -> QueryCase:
    source = manifest["meshes"][row["uid"]]
    with np.load(source["cache_path"]) as payload:
        values = {int(part): value.copy() for part, value in zip(payload["ids"], payload["x"], strict=True)}
    return QueryCase(row["uid"], row["query"], values, frozenset(row["target_ids"]))


def graph_figure(row: dict, manifest: dict, output: Path) -> list[Path]:
    case = load_embedding_case(row, manifest)
    graph = build_knn_graph(case.values, GRAPH["k"], GRAPH["edge_feature"])
    if len(graph) > 80:
        distance = {item["part_id"]: item["distance"] for item in row["candidate_records"]}
        keep = set(sorted(graph, key=lambda node: (distance[node], node))[:80])
        graph = graph.subgraph(keep).copy()
    # The frozen k=50 graph is appropriate for inference but unreadable when
    # drawn directly.  Preserve every displayed node while showing a maximum-
    # similarity spanning tree plus each node's two strongest affinities.
    affinities = weighted_edges(graph, GRAPH["sigma_factor"], GRAPH["edge_feature"])
    display = nx.Graph()
    display.add_nodes_from(graph.nodes)
    weighted = nx.Graph()
    weighted.add_nodes_from(graph.nodes)
    weighted.add_weighted_edges_from((left, right, weight) for (left, right), weight in affinities.items())
    if weighted.number_of_edges():
        display.add_edges_from(nx.maximum_spanning_edges(weighted, data=False))
    for node in weighted.nodes:
        strongest = sorted(
            weighted.edges(node, data="weight"),
            key=lambda edge: (-float(edge[2]), min(edge[0], edge[1]), max(edge[0], edge[1])),
        )[:2]
        display.add_edges_from((left, right) for left, right, _ in strongest)
    selected = set(row["methods"]["magiccut"]["selected_ids"])
    position = nx.spring_layout(display, seed=37, weight=None)
    colors = [COLORS["target"] if node in case.target else COLORS["false_positive"] if node in selected else "#CED4DA" for node in display]
    sizes = [95 if node == case.query else 32 for node in display]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    nx.draw_networkx_edges(display, position, ax=ax, alpha=.28, width=.7, edge_color="#8C98A4")
    nx.draw_networkx_nodes(display, position, ax=ax, node_color=colors, node_size=sizes, linewidths=[1.8 if node == case.query else 0 for node in display], edgecolors=COLORS["ink"])
    ax.set_title(f"Graph neighborhood — {row['uid'][:8]} / query {row['query']}", loc="left", weight="bold")
    ax.axis("off")
    return save(fig, output, "graph_visualization")


def medium_renders(uid: str, wanted: set[int], manifest33: dict) -> dict[int, Image.Image]:
    result = {}
    archive = Path(manifest33["archives"][uid]["local_path"])
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            if not member.isfile():
                continue
            try:
                parsed = parse_render_name(member.name)
            except ValueError:
                continue
            if parsed.part_id in wanted and parsed.size == "medium" and parsed.part_id not in result:
                source = handle.extractfile(member)
                if source is not None:
                    result[parsed.part_id] = Image.open(source).convert("RGB").copy()
    return result


def example_panel(row: dict, manifest33: dict, output: Path, name: str) -> list[Path]:
    target = set(row["target_ids"])
    selected = set(row["methods"]["magiccut"]["selected_ids"])
    groups = [
        ("Query", [row["query"]], COLORS["target"]),
        ("True positives", sorted((target & selected) - {row["query"]})[:3], COLORS["target"]),
        ("False positives", sorted(selected - target)[:3], COLORS["false_positive"]),
        ("False negatives", sorted(target - selected)[:3], COLORS["false_negative"]),
    ]
    groups = [group for group in groups if group[1]]
    ids = {part for _, parts, _ in groups for part in parts}
    images = medium_renders(row["uid"], ids, manifest33)
    columns = max(1, max(len(parts) for _, parts, _ in groups))
    fig, axes = plt.subplots(len(groups), columns, figsize=(2.2 * columns, 1.8 * len(groups)), squeeze=False)
    for row_index, (title, parts, color) in enumerate(groups):
        for column in range(columns):
            ax = axes[row_index, column]
            ax.axis("off")
            if column < len(parts) and parts[column] in images:
                part = parts[column]
                ax.imshow(images[part])
                ax.set_title(str(part), fontsize=8, color=color, weight="bold")
            if column == 0:
                ax.text(-.08, .5, title, transform=ax.transAxes, ha="right", va="center", color=color, weight="bold", fontsize=8)
    fig.suptitle(f"{row['uid'][:8]} / query {row['query']} — MagicCut F1 {row['methods']['magiccut']['f1']:.4f}", x=.02, ha="left", weight="bold")
    return save(fig, output, name)


def main() -> None:
    report35 = read_json("outputs/locked_evaluation/report.json")
    raw_path = Path("outputs/locked_evaluation/raw_results.json")
    if sha256_file(raw_path) != report35["raw_results_sha256"]:
        raise ValueError("locked evaluation raw-result digest mismatch")
    raw = read_json(raw_path)
    heldout = [row for row in raw["rows"] if row["uid"] not in set(VAL_UIDS)]
    deltas = [row["methods"]["magiccut"]["f1"] - row["methods"]["material_magic_wand"]["f1"] for row in heldout]
    success = heldout[int(np.argmax(deltas))]
    failure = heldout[int(np.argmin(deltas))]
    output = Path("outputs/figures")
    output.mkdir(parents=True, exist_ok=True)
    style()
    files = []
    files += method_diagram(output)
    files += precision_recall(heldout, output)
    files += f1_clicks(report35, output)
    files += calibration(heldout, output)
    files += risk_coverage(heldout, output)
    manifest34 = read_json("outputs/embeddings/manifest.json")
    manifest33 = read_json("outputs/benchmark_preparation/manifest.json")
    files += graph_figure(success, manifest34, output)
    files += example_panel(success, manifest33, output, "success_example")
    files += example_panel(failure, manifest33, output, "failure_example")
    manifest = {
        "source_raw_results_sha256": report35["raw_results_sha256"],
        "heldout_query_count": len(heldout),
        "success_example": {"uid": success["uid"], "query": success["query"], "f1_delta": max(deltas)},
        "failure_example": {"uid": failure["uid"], "query": failure["query"], "f1_delta": min(deltas)},
        "style": {"font": "DejaVu Sans", "png_dpi": 320, "colors": COLORS},
        "files": [{"path": str(path), "sha256": sha256_file(path)} for path in files],
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
