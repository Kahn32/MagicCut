#!/usr/bin/env python3
"""Paired, mesh-clustered statistical analysis of locked locked evaluation results."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from magiccut.io import read_json, sha256_file, write_json
from magiccut.frozen import VAL_UIDS


METHODS = ("material_magic_wand", "adaptive", "calibrated", "magiccut", "active_3")
PAIRS = (
    ("magiccut", "material_magic_wand"),
    ("active_3", "material_magic_wand"),
    ("active_3", "magiccut"),
)


def final_active(row: dict, click: int = 3) -> dict:
    return next(
        (item for item in reversed(row["active_history"]) if item["click"] <= click),
        row["active_history"][-1],
    )


def ranking_ap(row: dict, method: str) -> float:
    records = [item for item in row["candidate_records"] if item["part_id"] != row["query"]]
    labels = np.asarray([item["is_target"] for item in records], dtype=np.int8)
    if labels.sum() == 0:
        return 1.0
    if method == "material_magic_wand":
        scores = [-item["distance"] for item in records]
    elif method == "adaptive":
        scores = [-item["normalized_distance"] for item in records]
    elif method == "calibrated":
        scores = [item["calibrated_probability"] for item in records]
    else:
        selected = set(
            row["methods"][method]["selected_ids"]
            if method == "magiccut"
            else final_active(row)["selected_ids"]
        )
        # Graph methods return a binary partition, not a native ranking.  This
        # explicit decision-aware ranking puts selected nodes first and uses the
        # frozen calibrated probability only to break ties inside each partition.
        scores = [int(item["part_id"] in selected) + 1e-6 * item["calibrated_probability"] for item in records]
    return float(average_precision_score(labels, np.asarray(scores, dtype=float)))


def enrich(rows: list[dict]) -> list[dict]:
    enriched = []
    for row in rows:
        metrics = {name: dict(row["methods"][name]) for name in METHODS[:-1]}
        metrics["active_3"] = dict(final_active(row))
        enriched.append(
            {
                "uid": row["uid"],
                "query": row["query"],
                "candidate_count": row["candidate_count"],
                "target_size": row["target_size"],
                "positive_fraction": row["target_size"] / row["candidate_count"],
                "metrics": {
                    name: {"f1": float(metrics[name]["f1"]), "ranking_ap": ranking_ap(row, name)}
                    for name in METHODS
                },
            }
        )
    return enriched


def paired_values(rows: list[dict], better: str, reference: str, metric: str) -> np.ndarray:
    return np.asarray(
        [row["metrics"][better][metric] - row["metrics"][reference][metric] for row in rows],
        dtype=float,
    )


def mesh_bootstrap(rows: list[dict], better: str, reference: str, metric: str, samples: int, seed: int) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["uid"]].append(row)
    uids = sorted(grouped)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        chosen = rng.choice(uids, size=len(uids), replace=True)
        replicate = [item for uid in chosen for item in grouped[str(uid)]]
        draws.append(float(paired_values(replicate, better, reference, metric).mean()))
    observed = float(paired_values(rows, better, reference, metric).mean())
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "estimate": observed,
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "evidence_of_direction_at_95pct": bool(low > 0 or high < 0),
        "bootstrap_unit": "mesh",
        "samples": samples,
    }


def quantile_groups(rows: list[dict], field: str) -> list[dict]:
    values = np.asarray([row[field] for row in rows], dtype=float)
    cuts = np.unique(np.quantile(values, [0, 1 / 3, 2 / 3, 1]))
    result = []
    for index, (low, high) in enumerate(zip(cuts[:-1], cuts[1:], strict=True)):
        subset = [
            row
            for row in rows
            if row[field] >= low and (row[field] <= high if index == len(cuts) - 2 else row[field] < high)
        ]
        if not subset:
            continue
        result.append(
            {
                "low": float(low),
                "high": float(high),
                "query_count": len(subset),
                "mesh_count": len({row["uid"] for row in subset}),
                "magiccut_minus_mmw_f1": float(
                    paired_values(subset, "magiccut", "material_magic_wand", "f1").mean()
                ),
                "active_3_minus_mmw_f1": float(
                    paired_values(subset, "active_3", "material_magic_wand", "f1").mean()
                ),
            }
        )
    return result


def analyze(rows: list[dict], samples: int = 5000, seed: int = 36) -> dict:
    enriched = enrich(rows)
    paired = []
    confidence_intervals = {}
    for better, reference in PAIRS:
        for metric in ("f1", "ranking_ap"):
            values = paired_values(enriched, better, reference, metric)
            key = f"{better}_minus_{reference}.{metric}"
            paired.append(
                {
                    "comparison": f"{better}_minus_{reference}",
                    "metric": metric,
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "improved_queries": int((values > 0).sum()),
                    "tied_queries": int((values == 0).sum()),
                    "regressed_queries": int((values < 0).sum()),
                }
            )
            confidence_intervals[key] = mesh_bootstrap(
                enriched, better, reference, metric, samples=samples, seed=seed
            )
    catastrophic = {}
    for method in METHODS:
        values = np.asarray([row["metrics"][method]["f1"] for row in enriched])
        catastrophic[method] = {
            "f1_zero": int((values == 0).sum()),
            "f1_at_most_0_20": int((values <= 0.20).sum()),
            "worst_f1": float(values.min()),
        }
    deltas = paired_values(enriched, "magiccut", "material_magic_wand", "f1")
    worst_indices = np.argsort(deltas)[:10]
    return {
        "query_count": len(enriched),
        "mesh_count": len({row["uid"] for row in enriched}),
        "ranking_ap_definition": {
            "direct_methods": "native frozen distance/probability ordering, excluding the query part",
            "graph_methods": "selected partition first, calibrated probability as a deterministic within-partition tie-break",
            "caveat": "Graph ranking AP is decision-aware and must not be described as a native continuous graph confidence score.",
        },
        "paired_query_level": paired,
        "mesh_bootstrap_95_ci": confidence_intervals,
        "by_mesh_size": quantile_groups(enriched, "candidate_count"),
        "by_group_size": quantile_groups(enriched, "target_size"),
        "by_class_imbalance": quantile_groups(enriched, "positive_fraction"),
        "catastrophic_failures": catastrophic,
        "ten_largest_magiccut_regressions": [
            {
                "uid": enriched[index]["uid"],
                "query": enriched[index]["query"],
                "f1_delta": float(deltas[index]),
                "mmw_f1": enriched[index]["metrics"]["material_magic_wand"]["f1"],
                "magiccut_f1": enriched[index]["metrics"]["magiccut"]["f1"],
            }
            for index in worst_indices
        ],
    }


def main() -> None:
    prompt35 = read_json("reports/generated/locked_evaluation/report.json")
    raw_path = Path("reports/generated/locked_evaluation/raw_results.json")
    if sha256_file(raw_path) != prompt35["raw_results_sha256"]:
        raise ValueError("Immutable locked evaluation digest mismatch")
    raw = read_json(raw_path)
    rows = raw["rows"]
    heldout = [row for row in rows if row["uid"] not in set(prompt35["locked_config"].get("validation_uids", []))]
    # Older locked configs intentionally omit the UID list; recover it from the
    # evaluator's six fixed validation meshes using the exact full/held-out counts.
    if len(heldout) == len(rows):
        heldout = [row for row in rows if row["uid"] not in set(VAL_UIDS)]
    report = {
        "source_raw_results_sha256": prompt35["raw_results_sha256"],
        "full_100_meshes": analyze(rows),
        "heldout_94_meshes": analyze(heldout),
        "claim_policy": "No statistical-significance claim is made. Directional evidence is reported only when the mesh-bootstrap 95% interval excludes zero.",
    }
    output = Path("reports/generated/statistics")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", report)
    with (output / "final_quantitative_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "comparison", "metric", "estimate", "ci_95_low", "ci_95_high"],
        )
        writer.writeheader()
        for split in ("full_100_meshes", "heldout_94_meshes"):
            for key, value in report[split]["mesh_bootstrap_95_ci"].items():
                comparison, metric = key.split(".")
                writer.writerow({"split": split, "comparison": comparison, "metric": metric, **{
                    name: value[name] for name in ("estimate", "ci_95_low", "ci_95_high")
                }})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
