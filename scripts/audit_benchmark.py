#!/usr/bin/env python3
"""Download lightweight benchmark metadata and validate its exact structure."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from magiccut.data.metadata import (
    benchmark_summary,
    load_queries,
    normalize_dedup_metadata,
)
from magiccut.io import read_json, write_json
from magiccut.resources import (
    DATASET,
    DEDUP_METADATA_PATH,
    FULL_METADATA_PATH,
    LABELS_PREFIX,
    UIDS_PATH,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-metadata", action="store_true")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/benchmark"))
    args = parser.parse_args()

    api = HfApi()
    files = sorted(api.list_repo_files(DATASET.repo_id, repo_type=DATASET.repo_type))
    if args.download_metadata:
        snapshot_download(
            repo_id=DATASET.repo_id,
            repo_type=DATASET.repo_type,
            local_dir=args.raw_root,
            allow_patterns=[
                UIDS_PATH,
                f"{LABELS_PREFIX}*.json",
                DEDUP_METADATA_PATH,
                FULL_METADATA_PATH,
            ],
        )

    uids_file = args.raw_root / UIDS_PATH
    labels_root = args.raw_root / LABELS_PREFIX
    file_groups = Counter(
        "labels"
        if path.startswith(LABELS_PREFIX)
        else "deduplicated_render_archives"
        if path.startswith("test-bench/val_renders/images/")
        else "full_render_archives"
        if path.startswith("test-bench/val_renders_full/images/")
        else "other"
        for path in files
    )
    report: dict[str, object] = {
        "repo_file_count": len(files),
        "repo_file_groups": dict(sorted(file_groups.items())),
        "metadata_downloaded": uids_file.exists() and labels_root.exists(),
    }
    if report["metadata_downloaded"]:
        uids = read_json(uids_file)
        queries = load_queries(labels_root)
        report["benchmark"] = benchmark_summary(uids, queries)
        dedup_path = args.raw_root / DEDUP_METADATA_PATH
        full_path = args.raw_root / FULL_METADATA_PATH
        report["dedup_metadata_exists"] = dedup_path.exists()
        report["full_metadata_exists"] = full_path.exists()
        if dedup_path.exists() and full_path.exists():
            dedup = normalize_dedup_metadata(read_json(dedup_path))
            full = normalize_dedup_metadata(read_json(full_path))
            report["render_metadata"] = {
                "deduplicated_mesh_count": len(dedup),
                "full_mesh_count": len(full),
                "deduplicated_representative_count": sum(map(len, dedup.values())),
                "full_representative_count": sum(map(len, full.values())),
                "representative_mappings_equal": dedup == full,
            }

    write_json("outputs/benchmark_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
