#!/usr/bin/env python3
"""Run the locked 100-mesh evaluation without test-time tuning."""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from magiccut.adaptive import QueryCase, distances, group_metrics
from magiccut.calibration import fit_calibrator, local_scale, predict_case_probabilities
from magiccut.data.metadata import load_queries, normalize_dedup_metadata, representative_for_part
from magiccut.dedup import collapse
from magiccut.feedback import FeedbackState, make_influence_chooser
from magiccut.frozen import (
    ADAPTIVE_THRESHOLD,
    DATASET_REVISION,
    DIRECT_THRESHOLD,
    GRAPH,
    INTERACTION,
    PROBABILITY_THRESHOLD,
    VAL_UIDS,
    locked_config,
)
from magiccut.graph import build_knn_graph, solve_graph_cut, unary_costs, weighted_edges
from magiccut.io import read_json, sha256_file, write_json, write_json_atomic
from magiccut.resources import CHECKPOINT_SHA256, DEDUP_METADATA_PATH, LABELS_PREFIX


def load_embeddings(manifest: dict) -> dict[str, dict[int, np.ndarray]]:
    embeddings = {}
    for uid, row in manifest["meshes"].items():
        if row.get("status") not in ("encoded", "reused"):
            raise ValueError(f"Incomplete embedding extraction cache: {uid}")
        path = Path(row["cache_path"])
        if sha256_file(path) != row["cache_sha256"]:
            raise ValueError(f"embedding extraction cache digest drift: {uid}")
        with np.load(path) as payload:
            ids = payload["ids"]
            x = payload["x"]
            if x.shape != tuple(row["x_shape"]) or not np.isfinite(x).all():
                raise ValueError(f"embedding extraction cache array drift: {uid}")
            embeddings[uid] = {int(part): value.copy() for part, value in zip(ids, x, strict=True)}
    return embeddings


def make_cases(uids, queries, groups, embeddings):
    cases = []
    wanted = set(uids)
    for item in queries:
        if item.uid not in wanted:
            continue
        mapping = groups[item.uid]
        query = representative_for_part(mapping, item.primary_query)
        target = frozenset(collapse(set(item.final_selection), mapping))
        cases.append(QueryCase(item.uid, query, embeddings[item.uid], target))
    return cases


def retrieval_metrics(case: QueryCase) -> dict[str, float]:
    ranked = [part for _, part in sorted((score, part) for part, score in distances(case).items() if part != case.query)]
    relevant = set(case.target) - {case.query}
    if not relevant:
        return {"average_precision": 1.0, "pr_auc": 1.0, "r_precision": 1.0, "recall_at_20": 1.0}
    hits = np.asarray([int(part in relevant) for part in ranked], dtype=float)
    cumulative = np.cumsum(hits)
    precision = cumulative / np.arange(1, len(hits) + 1)
    recall = cumulative / len(relevant)
    average_precision = float(np.sum(precision * hits) / len(relevant))
    pr_auc = float(np.trapezoid(np.r_[1.0, precision], np.r_[0.0, recall]))
    r = len(relevant)
    return {
        "average_precision": average_precision,
        "pr_auc": pr_auc,
        "r_precision": float(hits[:r].sum() / r),
        "recall_at_20": float(hits[:20].sum() / len(relevant)),
    }


def active(case, probabilities, pairwise):
    chooser = make_influence_chooser(
        tuple(INTERACTION["weights"][name] for name in ("uncertainty", "degree", "diversity")),
        INTERACTION["uncertainty_method"],
    )
    positive = {case.query}
    negative = set()
    history = []
    unaries = unary_costs(probabilities, PROBABILITY_THRESHOLD)
    for click in range(INTERACTION["click_budget"] + 1):
        started = time.perf_counter()
        result = solve_graph_cut(unaries, pairwise, GRAPH["pairwise_lambda"], positive, negative)
        row = {
            "click": click,
            **group_metrics(set(result.selected), set(case.target)),
            "selected_ids": sorted(map(int, result.selected)),
            "inference_ms": 1000 * (time.perf_counter() - started),
            "positive_constraints": sorted(positive),
            "negative_constraints": sorted(negative),
        }
        history.append(row)
        if click == INTERACTION["click_budget"] or row["f1"] == 1.0:
            break
        unlabeled = frozenset(set(case.values) - positive - negative)
        if not unlabeled:
            break
        state = FeedbackState(result.selected, frozenset(positive), frozenset(negative), unlabeled)
        started = time.perf_counter()
        node = int(
            chooser(
                case,
                probabilities,
                GRAPH | {"decision_threshold": PROBABILITY_THRESHOLD},
                state,
                None,
                pairwise,
                np.random.default_rng(0),
            )
        )
        row["acquisition_ms"] = 1000 * (time.perf_counter() - started)
        row["asked_part"] = node
        row["oracle_label"] = int(node in case.target)
        (positive if node in case.target else negative).add(node)
    return history


def mean_dict(rows, keys):
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def summarize(rows):
    methods = {}
    for method in ("material_magic_wand", "adaptive", "calibrated", "magiccut"):
        values = [row["methods"][method] for row in rows]
        methods[method] = mean_dict(values, ("precision", "recall", "f1", "latency_ms"))
    retrieval = mean_dict([row["retrieval"] for row in rows], ("average_precision", "pr_auc", "r_precision", "recall_at_20"))
    interaction = []
    for click in range(INTERACTION["click_budget"] + 1):
        values = []
        for row in rows:
            values.append(next((item for item in reversed(row["active_history"]) if item["click"] <= click), row["active_history"][-1]))
        interaction.append({"click": click, **mean_dict(values, ("precision", "recall", "f1", "inference_ms"))})
    return {"query_count": len(rows), "mesh_count": len({row["uid"] for row in rows}), "methods": methods, "retrieval": retrieval, "active": interaction}


def flat_metrics(rows):
    summary = summarize(rows)
    result = {}
    for method, values in summary["methods"].items():
        for metric, value in values.items():
            result[f"{method}.{metric}"] = value
    for metric, value in summary["retrieval"].items():
        result[f"retrieval.{metric}"] = value
    for row in summary["active"]:
        for metric in ("precision", "recall", "f1", "inference_ms"):
            result[f"active.click_{row['click']}.{metric}"] = row[metric]
    return result


def bootstrap_by_mesh(rows, samples=2000, seed=0):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["uid"]].append(row)
    uids = sorted(grouped)
    rng = np.random.default_rng(seed)
    observed = flat_metrics(rows)
    draws = {key: [] for key in observed}
    for _ in range(samples):
        sampled = rng.choice(uids, size=len(uids), replace=True)
        replicate = [row for uid in sampled for row in grouped[str(uid)]]
        values = flat_metrics(replicate)
        for key, value in values.items():
            draws[key].append(value)
    return {
        key: {
            "estimate": value,
            "ci_95_low": float(np.percentile(draws[key], 2.5)),
            "ci_95_high": float(np.percentile(draws[key], 97.5)),
        }
        for key, value in observed.items()
    }


def verify_frozen_inputs():
    graph = read_json("reports/generated/graph_sweep/frozen_graph_configuration.json")
    frozen = read_json("reports/generated/interaction_experiments/frozen_interactive_configuration.json")
    report = read_json("reports/generated/interaction_experiments/report.json")
    tuned = report["prompt_21"]["tuned_on_validation"]
    expected_weights = INTERACTION["weights"]
    if graph != GRAPH or frozen["graph"] != GRAPH:
        raise ValueError("Frozen graph configuration drift")
    if frozen["uncertainty_method"] != INTERACTION["uncertainty_method"] or frozen["clarification_policy"] != INTERACTION["clarification_policy"]:
        raise ValueError("Frozen interaction policy drift")
    if frozen["influence_weights"] != expected_weights or frozen["click_budget"] != INTERACTION["click_budget"]:
        raise ValueError("Frozen interaction parameters drift")
    if not np.isclose(frozen["probability_threshold"], PROBABILITY_THRESHOLD):
        raise ValueError("Frozen probability threshold drift")
    if not np.isclose(tuned["direct_distance_threshold"], DIRECT_THRESHOLD) or not np.isclose(tuned["adaptive_normalized_threshold"], ADAPTIVE_THRESHOLD):
        raise ValueError("Frozen baseline threshold drift")
    if not report["prompt_21"]["graph_gate"]["passed"]:
        raise ValueError("MagicCut did not pass Gate 2")


def main():
    output = Path("reports/generated/locked_evaluation")
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw_results.json"
    if raw_path.exists():
        raise FileExistsError("locked evaluation raw results are immutable; refusing to overwrite")
    verify_frozen_inputs()
    manifest = read_json("reports/generated/embeddings/manifest.json")
    if not manifest.get("all_100_complete") or manifest.get("failures") or manifest.get("completed_meshes") != 100:
        raise ValueError("embedding extraction is not complete and clean")
    if manifest["checkpoint_sha256"] != CHECKPOINT_SHA256 or manifest["dataset_revision"] != DATASET_REVISION:
        raise ValueError("embedding extraction provenance drift")

    benchmark = Path("data/raw/benchmark")
    groups = normalize_dedup_metadata(read_json(benchmark / DEDUP_METADATA_PATH))
    queries = load_queries(benchmark / LABELS_PREFIX)
    embeddings = load_embeddings(manifest)
    all_uids = sorted(embeddings)
    cases = make_cases(all_uids, queries, groups, embeddings)
    validation_cases = make_cases(VAL_UIDS, queries, groups, embeddings)
    calibrator = fit_calibrator(validation_cases)
    probabilities = predict_case_probabilities(calibrator, cases)
    progress_path = output / "progress.json"
    progress = read_json(progress_path) if progress_path.exists() else {"locked_config": locked_config(), "rows": []}
    if progress["locked_config"] != locked_config():
        raise ValueError("locked evaluation progress configuration drift")
    done = {(row["uid"], row["query"]) for row in progress["rows"]}
    graph_cache = {}

    for index, (case, probs) in enumerate(zip(cases, probabilities, strict=True), 1):
        key = (case.uid, case.query)
        if key in done:
            print(f"[{index:03d}/{len(cases)}] {case.uid}/{case.query} cached", flush=True)
            continue
        if case.uid not in graph_cache:
            graph = build_knn_graph(case.values, GRAPH["k"], GRAPH["edge_feature"])
            graph_cache[case.uid] = weighted_edges(graph, GRAPH["sigma_factor"], GRAPH["edge_feature"])
        pairwise = graph_cache[case.uid]
        d = distances(case)
        methods = {}
        selectors = (
            ("material_magic_wand", lambda: {part for part, value in d.items() if value <= DIRECT_THRESHOLD} | {case.query}),
            ("adaptive", lambda: {part for part, value in d.items() if value / local_scale(case) <= ADAPTIVE_THRESHOLD} | {case.query}),
            ("calibrated", lambda: {part for part, value in probs.items() if value >= PROBABILITY_THRESHOLD}),
        )
        for name, selector in selectors:
            started = time.perf_counter()
            selected = selector()
            methods[name] = group_metrics(set(selected), set(case.target)) | {
                "latency_ms": 1000 * (time.perf_counter() - started),
                "selected_ids": sorted(map(int, selected)),
            }
        started = time.perf_counter()
        cut = solve_graph_cut(unary_costs(probs, PROBABILITY_THRESHOLD), pairwise, GRAPH["pairwise_lambda"], {case.query}, set())
        methods["magiccut"] = group_metrics(set(cut.selected), set(case.target)) | {
            "latency_ms": 1000 * (time.perf_counter() - started),
            "selected_ids": sorted(map(int, cut.selected)),
        }
        scale = local_scale(case)
        weighted_degree = defaultdict(float)
        for (left, right), weight in pairwise.items():
            weighted_degree[int(left)] += float(weight)
            weighted_degree[int(right)] += float(weight)
        candidate_records = [
            {
                "part_id": int(part),
                "is_target": bool(part in case.target),
                "distance": float(d[part]),
                "normalized_distance": float(d[part] / scale),
                "calibrated_probability": float(probs[part]),
                "weighted_graph_degree": float(weighted_degree[part]),
            }
            for part in sorted(case.values)
        ]
        row = {
            "uid": case.uid,
            "query": case.query,
            "candidate_count": len(case.values),
            "target_size": len(case.target),
            "target_ids": sorted(map(int, case.target)),
            "candidate_records": candidate_records,
            "retrieval": retrieval_metrics(case),
            "methods": methods,
            "active_history": active(case, probs, pairwise),
        }
        progress["rows"].append(row)
        write_json_atomic(progress_path, progress)
        print(f"[{index:03d}/{len(cases)}] {case.uid}/{case.query} evaluated", flush=True)

    rows = progress["rows"]
    heldout_rows = [row for row in rows if row["uid"] not in set(VAL_UIDS)]
    if len(rows) != 241 or len({row["uid"] for row in rows}) != 100 or len({row["uid"] for row in heldout_rows}) != 94:
        raise ValueError("locked evaluation full/held-out coverage mismatch")
    report = {
        "locked_config": locked_config(),
        "full_100_meshes": summarize(rows),
        "heldout_94_meshes": summarize(heldout_rows),
        "bootstrap_95_ci_by_mesh": {
            "full_100_meshes": bootstrap_by_mesh(rows),
            "heldout_94_meshes": bootstrap_by_mesh(heldout_rows),
        },
        "raw_results_sha256": None,
    }
    payload = {"locked_config": locked_config(), "rows": rows}
    write_json_atomic(raw_path, payload)
    report["raw_results_sha256"] = sha256_file(raw_path)
    write_json(output / "report.json", report)
    os.chmod(raw_path, 0o444)
    write_json(output / "completion.json", {"status": "complete", "raw_results_sha256": report["raw_results_sha256"]})
    print(json.dumps({"full_100_meshes": report["full_100_meshes"], "heldout_94_meshes": report["heldout_94_meshes"]}, indent=2))


if __name__ == "__main__":
    main()
