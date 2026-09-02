import numpy as np

from scripts.analyze_statistics import enrich, mesh_bootstrap, ranking_ap


def row(uid="mesh", query=1, mmw=(1, 2), cut=(1, 2, 3), active=(1, 2)):
    return {
        "uid": uid,
        "query": query,
        "candidate_count": 4,
        "target_size": 2,
        "candidate_records": [
            {"part_id": 1, "is_target": True, "distance": 0.0, "normalized_distance": 0.0, "calibrated_probability": 1.0},
            {"part_id": 2, "is_target": True, "distance": 1.0, "normalized_distance": 0.1, "calibrated_probability": 0.9},
            {"part_id": 3, "is_target": False, "distance": 2.0, "normalized_distance": 0.2, "calibrated_probability": 0.8},
            {"part_id": 4, "is_target": False, "distance": 3.0, "normalized_distance": 0.3, "calibrated_probability": 0.1},
        ],
        "methods": {
            "material_magic_wand": {"f1": 1.0, "selected_ids": list(mmw)},
            "adaptive": {"f1": 1.0, "selected_ids": list(mmw)},
            "calibrated": {"f1": 1.0, "selected_ids": list(mmw)},
            "magiccut": {"f1": 0.8, "selected_ids": list(cut)},
        },
        "active_history": [{"click": 0, "f1": 0.8, "selected_ids": list(cut)}, {"click": 1, "f1": 1.0, "selected_ids": list(active)}],
    }


def test_ranking_ap_excludes_query_and_preserves_order():
    item = row()
    assert ranking_ap(item, "material_magic_wand") == 1.0
    assert ranking_ap(item, "calibrated") == 1.0
    assert ranking_ap(item, "magiccut") <= 1.0


def test_mesh_bootstrap_is_deterministic_and_paired():
    rows = enrich([row("a"), row("b", mmw=(1, 2, 3), cut=(1, 2), active=(1, 2))])
    first = mesh_bootstrap(rows, "active_3", "material_magic_wand", "f1", samples=100, seed=7)
    second = mesh_bootstrap(rows, "active_3", "material_magic_wand", "f1", samples=100, seed=7)
    assert first == second
    expected = np.mean([item["metrics"]["active_3"]["f1"] - item["metrics"]["material_magic_wand"]["f1"] for item in rows])
    assert first["estimate"] == expected
