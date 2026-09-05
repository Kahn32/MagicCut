#!/usr/bin/env python3
"""Real-data acceptance audit for the interactive demo interactive demo."""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from magiccut.demo import MagicCutDemo
from magiccut.io import read_json, sha256_file, write_json
import serve_demo as server


ROOT = Path(__file__).resolve().parents[1]
CASE_UID = "8186e08947664545be29caae9c9ff01c"
CASE_QUERY = 189


def require_static_contract() -> dict:
    html = (ROOT / "demo/index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "demo/app.js").read_text(encoding="utf-8")
    css = (ROOT / "demo/style.css").read_text(encoding="utf-8").lower()
    required_ids = ("mesh", "query", "run", "reset", "clicks", "confidence", "yes", "no", "parts")
    missing_ids = [item for item in required_ids if f'id="{item}"' not in html]
    endpoint_contract = ("/api/meshes", "/api/start", "/api/feedback", "/api/reset", "/api/render/")
    missing_endpoints = [item for item in endpoint_contract if item not in javascript]
    colors = {
        "query_red": "--red:#d94b4b" in css,
        "selected_orange": "--orange:#ee8b2d" in css,
        "uncertain_yellow": "--yellow:#f2cf55" in css,
        "recommended_blue": "--blue:#246bce" in css,
    }
    if missing_ids or missing_endpoints or not all(colors.values()):
        raise AssertionError({"missing_ids": missing_ids, "missing_endpoints": missing_endpoints, "colors": colors})
    return {"required_controls": list(required_ids), "required_endpoints": list(endpoint_contract), "colors": colors}


def main() -> None:
    prompt35 = read_json(ROOT / "outputs/locked_evaluation/report.json")
    raw_path = ROOT / "outputs/locked_evaluation/raw_results.json"
    if sha256_file(raw_path) != prompt35["raw_results_sha256"]:
        raise ValueError("locked evaluation digest mismatch")
    raw = read_json(raw_path)
    row = next(item for item in raw["rows"] if item["uid"] == CASE_UID and item["query"] == CASE_QUERY)
    target = set(row["target_ids"])

    began = time.perf_counter()
    engine = MagicCutDemo(ROOT)
    load_ms = 1000 * (time.perf_counter() - began)
    meshes = engine.meshes()
    if len(meshes) != 100 or sum(item["part_count"] for item in meshes) != 25_348:
        raise AssertionError("Demo feature coverage mismatch")

    server.engine = engine
    client = TestClient(server.app)
    statuses = {}
    began = time.perf_counter()
    response = client.post("/api/start", json={"uid": CASE_UID, "query_part": CASE_QUERY})
    statuses["start"] = response.status_code
    state = response.json()
    start_ms = 1000 * (time.perf_counter() - began)
    universe = set(next(item for item in meshes if item["uid"] == CASE_UID)["part_ids"])
    partition = set(state["confident_positive"]) | set(state["uncertain"]) | set(state["confident_negative"])
    if partition != universe or state["query_part"] != CASE_QUERY or not state["baseline_selected"]:
        raise AssertionError("Demo start-state contract mismatch")

    histories = [{"click": 0, "recommended_part": state["recommended_part"], "selected": len(state["selected"])}]
    feedback_ms = []
    for click in range(1, 4):
        part = state["recommended_part"]
        if part is None:
            break
        began = time.perf_counter()
        response = client.post(
            "/api/feedback",
            json={"session_id": state["session_id"], "part_id": part, "positive": part in target},
        )
        feedback_ms.append(1000 * (time.perf_counter() - began))
        statuses[f"feedback_{click}"] = response.status_code
        state = response.json()
        histories.append({"click": click, "recommended_part": state["recommended_part"], "selected": len(state["selected"])})

    render = client.get(f"/api/render/{CASE_UID}/{CASE_QUERY}")
    statuses["render"] = render.status_code
    if render.status_code != 200 or render.headers.get("content-type") != "image/png" or len(render.content) < 100:
        raise AssertionError("Part-render endpoint contract mismatch")
    reset = client.post("/api/reset", json={"session_id": state["session_id"]})
    statuses["reset"] = reset.status_code
    reset_state = reset.json()
    if reset_state["click_count"] != 0 or reset_state["positive_constraints"] != [CASE_QUERY] or reset_state["negative_constraints"]:
        raise AssertionError("Demo reset contract mismatch")
    if any(status != 200 for status in statuses.values()):
        raise AssertionError(statuses)

    report = {
        "source_raw_results_sha256": prompt35["raw_results_sha256"],
        "mesh_count": len(meshes),
        "representative_count": sum(item["part_count"] for item in meshes),
        "acceptance_case": {"uid": CASE_UID, "query": CASE_QUERY, "candidate_count": row["candidate_count"]},
        "backend": {
            "load_ms": load_ms,
            "start_roundtrip_ms": start_ms,
            "feedback_roundtrip_ms": feedback_ms,
            "http_statuses": statuses,
            "render_bytes": len(render.content),
            "partition_complete": True,
            "reset_verified": True,
            "interaction_history": histories,
        },
        "interface": require_static_contract(),
    }
    output = ROOT / "outputs/demo_validation"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
