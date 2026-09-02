import json
from pathlib import Path

from magiccut.data.metadata import (
    benchmark_summary,
    load_query,
    normalize_dedup_metadata,
    representative_for_part,
)


def test_load_query(tmp_path: Path) -> None:
    query_file = tmp_path / "mesh-a" / "7_final.json"
    query_file.parent.mkdir()
    query_file.write_text(
        '{"material_id": 7, "primary_query": 3, "final_selection": [3, 9]}',
        encoding="utf-8",
    )
    query = load_query(query_file)
    assert query.uid == "mesh-a"
    assert query.primary_query == 3
    assert query.final_selection == (3, 9)


def test_load_query_allows_release_anomaly_missing_material_id(tmp_path: Path) -> None:
    query_file = tmp_path / "mesh-a" / "3_final.json"
    query_file.parent.mkdir()
    query_file.write_text(
        json.dumps({"primary_query": 3, "final_selection": [3, 9]}),
        encoding="utf-8",
    )
    assert load_query(query_file).material_id is None


def test_benchmark_summary() -> None:
    class Query:
        uid = "mesh-a"
        primary_query = 3
        final_selection = (3, 9)
        original_selection = None
        material_id = 7

    result = benchmark_summary(["mesh-a"], [Query()])
    assert result["mesh_count"] == 1
    assert result["query_count"] == 1
    assert result["primary_query_in_final_count"] == 1


def test_normalize_released_dedup_schema() -> None:
    normalized = normalize_dedup_metadata(
        {
            "mesh": {
                "total_components": 4,
                "unique_components": 2,
                "unique_ids": {"0": [1], "2": [3]},
                "material_mapping": [0, 1],
            }
        }
    )
    assert normalized == {"mesh": {0: (1,), 2: (3,)}}
    assert representative_for_part(normalized["mesh"], 3) == 2
