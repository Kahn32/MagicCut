#!/usr/bin/env python3
"""Download, securely extract, and audit one deduplicated render archive."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from magiccut.data.metadata import (
    load_queries,
    normalize_dedup_metadata,
    representative_for_part,
)
from magiccut.data.renders import (
    create_contact_sheet,
    inspect_render_files,
    parse_render_name,
    safe_extract_tar,
)
from magiccut.io import read_json, write_json
from magiccut.resources import (
    DATASET,
    DEDUP_IMAGES_PREFIX,
    DEDUP_METADATA_PATH,
    LABELS_PREFIX,
)


def archive_candidates(files: list[str]) -> list[str]:
    return sorted(
        path for path in files
        if path.startswith(DEDUP_IMAGES_PREFIX) and path.endswith(".tar.gz")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uid")
    parser.add_argument("--select-smallest", action="store_true")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/benchmark"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/one_mesh"))
    args = parser.parse_args()
    if bool(args.uid) == bool(args.select_smallest):
        parser.error("choose exactly one of --uid or --select-smallest")

    api = HfApi()
    files = sorted(api.list_repo_files(DATASET.repo_id, repo_type=DATASET.repo_type))
    archives = archive_candidates(files)
    if not archives:
        raise RuntimeError("No deduplicated render archives found")

    if args.select_smallest:
        details = api.list_repo_tree(
            DATASET.repo_id,
            path_in_repo=DEDUP_IMAGES_PREFIX.rstrip("/"),
            repo_type=DATASET.repo_type,
            recursive=False,
            expand=True,
        )
        sized = sorted(
            (entry.size, entry.path)
            for entry in details
            if getattr(entry, "path", "").endswith(".tar.gz")
        )
        if not sized:
            raise RuntimeError("Could not obtain archive sizes")
        archive_size, archive_path = sized[0]
        uid = Path(archive_path).name.removesuffix(".tar.gz")
    else:
        uid = args.uid
        archive_path = f"{DEDUP_IMAGES_PREFIX}{uid}.tar.gz"
        if archive_path not in archives:
            raise ValueError(f"No deduplicated archive found for {uid}")
        archive_size = None

    local_archive = hf_hub_download(
        repo_id=DATASET.repo_id,
        repo_type=DATASET.repo_type,
        filename=archive_path,
        local_dir=args.raw_root,
    )
    extracted = safe_extract_tar(local_archive, args.cache_root / uid)
    renders = inspect_render_files(extracted)
    queries = [
        query for query in load_queries(args.raw_root / LABELS_PREFIX) if query.uid == uid
    ]
    metadata = normalize_dedup_metadata(
        read_json(args.raw_root / DEDUP_METADATA_PATH)
    )
    representatives = metadata[uid]
    full_render_by_part = {
        parsed.part_id: path
        for path in extracted
        if (parsed := parse_render_name(path)).size == "full"
    }
    query_rows = []
    query_details = []
    for query in queries:
        target_representatives = sorted(
            {
                representative_for_part(representatives, part_id)
                for part_id in query.final_selection
            }
        )
        query_representative = representative_for_part(
            representatives, query.primary_query
        )
        shown = [query_representative] + [
            part_id for part_id in target_representatives if part_id != query_representative
        ]
        missing = [part_id for part_id in shown if part_id not in full_render_by_part]
        if missing:
            raise ValueError(f"Missing full renders for representative ids: {missing}")
        query_rows.append(
            (
                f"query {query.primary_query}",
                [
                    (
                        ("query" if part_id == query_representative else "target")
                        + f" rep {part_id}",
                        full_render_by_part[part_id],
                    )
                    for part_id in shown
                ],
            )
        )
        query_details.append(
            {
                "primary_query": query.primary_query,
                "query_representative": query_representative,
                "final_selection": list(query.final_selection),
                "target_representatives": target_representatives,
            }
        )
    contact_sheet = create_contact_sheet(
        query_rows, "reports/generated/one_mesh_contact_sheet.png"
    )
    sizes = Counter(item["size"] for item in renders)
    views = Counter(item["view"] for item in renders)
    part_ids = sorted({int(item["part_id"]) for item in renders})
    primary_queries = sorted(query.primary_query for query in queries)
    report = {
        "uid": uid,
        "archive_path": archive_path,
        "reported_archive_size": archive_size,
        "downloaded_archive_size": Path(local_archive).stat().st_size,
        "render_count": len(renders),
        "render_sizes": dict(sorted(sizes.items())),
        "render_views": dict(sorted(views.items())),
        "unique_rendered_part_ids": len(part_ids),
        "query_count": len(queries),
        "primary_queries": primary_queries,
        "primary_queries_with_render": sorted(set(primary_queries) & set(part_ids)),
        "primary_queries_without_render": sorted(set(primary_queries) - set(part_ids)),
        "query_details": query_details,
        "contact_sheet": str(contact_sheet),
        "sample_renders": renders[:20],
    }
    write_json("reports/generated/one_mesh_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
