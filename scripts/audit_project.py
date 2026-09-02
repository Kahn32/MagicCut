#!/usr/bin/env python3
"""Verify the final artifact chain and write a completion manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports/generated"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    raw = REPORTS / "locked_evaluation/raw_results.json"
    p35 = load(REPORTS / "locked_evaluation/report.json")
    raw_sha = digest(raw)
    assert raw_sha == p35["raw_results_sha256"]
    assert p35["full_100_meshes"]["mesh_count"] == 100
    assert p35["full_100_meshes"]["query_count"] == 241
    assert p35["heldout_94_meshes"]["mesh_count"] == 94
    assert p35["heldout_94_meshes"]["query_count"] == 230

    p36 = load(REPORTS / "statistics/report.json")
    assert p36["source_raw_results_sha256"] == raw_sha
    assert (REPORTS / "statistics/final_quantitative_table.csv").is_file()

    p37 = load(REPORTS / "figures/manifest.json")
    assert p37["source_raw_results_sha256"] == raw_sha
    assert len(p37["files"]) == 16
    for item in p37["files"]:
        path = ROOT / item["path"]
        assert path.is_file() and digest(path) == item["sha256"]

    p3839 = load(REPORTS / "demo_validation/report.json")
    assert p3839["source_raw_results_sha256"] == raw_sha
    assert p3839["mesh_count"] == 100
    assert p3839["representative_count"] == 25_348
    assert all(value is True for value in p3839["backend"].values() if isinstance(value, bool))
    assert all(value is True for value in p3839["interface"]["colors"].values())
    assert len(p3839["interface"]["required_controls"]) == 9
    assert len(p3839["interface"]["required_endpoints"]) == 5
    assert all(status == 200 for status in p3839["backend"]["http_statuses"].values())

    required_docs = [
        ROOT / "README.md",
        ROOT / "docs/research_writeup.md",
        ROOT / "docs/technical_summary.md",
        ROOT / "docs/three_minute_demo_script.md",
        ROOT / "docs/outreach_package.md",
    ]
    claims = load(REPORTS / "presentation/numerical_claim_audit.json")
    assert claims["status"] == "passed"
    assert claims["source_raw_results_sha256"] == raw_sha
    for path in required_docs:
        assert path.is_file() and path.stat().st_size > 0

    demo = load(REPORTS / "presentation/demo_manifest.json")
    demo_path = ROOT / demo["demo_gif"]
    assert demo["source_raw_results_sha256"] == raw_sha
    assert demo["frame_count"] == 4
    assert digest(demo_path) == demo["demo_gif_sha256"]

    stages = {
        "locked_evaluation": "passed",
        "statistical_analysis": "passed",
        "research_figures": "passed",
        "demo_backend": "passed",
        "demo_interface": "passed",
        "reproducibility_cleanup": "passed",
        "research_writeup": "passed",
        "project_presentation": "passed",
    }
    completion = {
        "status": "passed",
        "stages": stages,
        "source_raw_results_sha256": raw_sha,
        "coverage": {"meshes": 100, "queries": 241, "representative_parts": 25_348},
        "heldout": {
            "meshes": 94,
            "queries": 230,
            "material_magic_wand_f1": p35["heldout_94_meshes"]["methods"]["material_magic_wand"]["f1"],
            "magiccut_f1": p35["heldout_94_meshes"]["methods"]["magiccut"]["f1"],
            "magiccut_3_click_f1": p35["heldout_94_meshes"]["active"][3]["f1"],
        },
        "success_criteria_met": False,
        "interpretation": "Active feedback partially repairs graph-cut errors but does not recover the direct-retrieval baseline.",
    }
    output = REPORTS / "completion/completion.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
