"""Stateful backend used by the local MagicCut interaction demo."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from magiccut.adaptive import QueryCase, distances
from magiccut.calibration import fit_calibrator, predict_case_probabilities
from magiccut.data.metadata import load_queries, normalize_dedup_metadata, representative_for_part
from magiccut.dedup import collapse
from magiccut.feedback import FeedbackState, make_influence_chooser
from magiccut.frozen import (
    DIRECT_THRESHOLD,
    GRAPH,
    INTERACTION,
    PROBABILITY_THRESHOLD,
    VAL_UIDS,
)
from magiccut.graph import build_knn_graph, solve_graph_cut, unary_costs, weighted_edges
from magiccut.io import read_json, sha256_file
from magiccut.resources import DEDUP_METADATA_PATH, LABELS_PREFIX


@dataclass
class DemoSession:
    session_id: str
    case: QueryCase
    probabilities: dict[int, float]
    pairwise: dict[tuple[int, int], float]
    baseline: frozenset[int]
    positive: set[int] = field(default_factory=set)
    negative: set[int] = field(default_factory=set)
    clicks: int = 0


def _load_embeddings(manifest: dict) -> dict[str, dict[int, np.ndarray]]:
    if not manifest.get("all_100_complete") or manifest.get("completed_meshes") != 100 or manifest.get("failures"):
        raise ValueError("The demo requires a complete, clean embedding extraction manifest")
    result = {}
    for uid, row in manifest["meshes"].items():
        path = Path(row["cache_path"])
        if sha256_file(path) != row["cache_sha256"]:
            raise ValueError(f"Embedding digest drift for {uid}")
        with np.load(path) as payload:
            ids, values = payload["ids"], payload["x"]
            result[uid] = {int(part): value.copy() for part, value in zip(ids, values, strict=True)}
    return result


class MagicCutDemo:
    """Loads frozen features once and manages deterministic feedback sessions."""

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root)
        benchmark = self.root / "data/raw/benchmark"
        manifest = read_json(self.root / "outputs/embeddings/manifest.json")
        self.embeddings = _load_embeddings(manifest)
        self.groups = normalize_dedup_metadata(read_json(benchmark / DEDUP_METADATA_PATH))
        queries = load_queries(benchmark / LABELS_PREFIX)
        validation = []
        for item in queries:
            if item.uid not in set(VAL_UIDS):
                continue
            mapping = self.groups[item.uid]
            validation.append(
                QueryCase(
                    item.uid,
                    representative_for_part(mapping, item.primary_query),
                    self.embeddings[item.uid],
                    frozenset(collapse(set(item.final_selection), mapping)),
                )
            )
        self.calibrator = fit_calibrator(validation)
        self.sessions: dict[str, DemoSession] = {}
        self.chooser = make_influence_chooser(
            tuple(INTERACTION["weights"][name] for name in ("uncertainty", "degree", "diversity")),
            INTERACTION["uncertainty_method"],
        )

    def meshes(self) -> list[dict]:
        return [
            {"uid": uid, "part_count": len(values), "part_ids": sorted(values)}
            for uid, values in sorted(self.embeddings.items(), key=lambda item: (len(item[1]), item[0]))
        ]

    def start(self, uid: str, query_part: int) -> dict:
        if uid not in self.embeddings:
            raise KeyError(f"Unknown mesh: {uid}")
        if query_part not in self.embeddings[uid]:
            raise KeyError(f"Unknown representative part {query_part} for {uid}")
        case = QueryCase(uid, int(query_part), self.embeddings[uid], frozenset())
        probabilities = predict_case_probabilities(self.calibrator, [case])[0]
        graph = build_knn_graph(case.values, GRAPH["k"], GRAPH["edge_feature"])
        pairwise = weighted_edges(graph, GRAPH["sigma_factor"], GRAPH["edge_feature"])
        direct_distances = distances(case)
        baseline = frozenset(part for part, value in direct_distances.items() if value <= DIRECT_THRESHOLD) | {
            case.query
        }
        session_id = uuid.uuid4().hex
        session = DemoSession(
            session_id=session_id,
            case=case,
            probabilities=probabilities,
            pairwise=pairwise,
            baseline=baseline,
            positive={case.query},
        )
        self.sessions[session_id] = session
        return self._result(session)

    def feedback(self, session_id: str, part_id: int, positive: bool) -> dict:
        session = self.sessions[session_id]
        part_id = int(part_id)
        if part_id not in session.case.values:
            raise KeyError(f"Unknown part {part_id}")
        if part_id == session.case.query:
            raise ValueError("The query part is a fixed positive constraint")
        session.positive.discard(part_id)
        session.negative.discard(part_id)
        (session.positive if positive else session.negative).add(part_id)
        session.clicks += 1
        return self._result(session)

    def reset(self, session_id: str) -> dict:
        session = self.sessions[session_id]
        session.positive = {session.case.query}
        session.negative.clear()
        session.clicks = 0
        return self._result(session)

    def _result(self, session: DemoSession) -> dict:
        started = time.perf_counter()
        cut = solve_graph_cut(
            unary_costs(session.probabilities, PROBABILITY_THRESHOLD),
            session.pairwise,
            GRAPH["pairwise_lambda"],
            session.positive,
            session.negative,
        )
        selected = set(cut.selected)
        unlabeled = set(session.case.values) - session.positive - session.negative
        state = FeedbackState(
            cut.selected,
            frozenset(session.positive),
            frozenset(session.negative),
            frozenset(unlabeled),
        )
        recommended = None
        if unlabeled:
            recommended = int(
                self.chooser(
                    session.case,
                    session.probabilities,
                    GRAPH | {"decision_threshold": PROBABILITY_THRESHOLD},
                    state,
                    None,
                    session.pairwise,
                    np.random.default_rng(0),
                )
            )
        uncertainty = {
            part: 1.0 - abs(2.0 * probability - 1.0)
            for part, probability in session.probabilities.items()
        }
        uncertain = {part for part, value in uncertainty.items() if value >= 0.5} - session.positive - session.negative
        confident_positive = selected - uncertain
        confident_negative = set(session.case.values) - selected - uncertain
        return {
            "session_id": session.session_id,
            "uid": session.case.uid,
            "query_part": session.case.query,
            "click_count": session.clicks,
            "baseline_selected": sorted(session.baseline),
            "selected": sorted(selected),
            "confident_positive": sorted(confident_positive),
            "uncertain": sorted(uncertain),
            "confident_negative": sorted(confident_negative),
            "recommended_part": recommended,
            "positive_constraints": sorted(session.positive),
            "negative_constraints": sorted(session.negative),
            "probabilities": {str(part): float(value) for part, value in session.probabilities.items()},
            "uncertainties": {str(part): float(value) for part, value in uncertainty.items()},
            "inference_ms": 1000 * (time.perf_counter() - started),
        }
