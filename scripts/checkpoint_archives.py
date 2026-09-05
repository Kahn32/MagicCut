#!/usr/bin/env python3
"""Create bounded, hash-audited recovery chunks for benchmark preparation raw inputs."""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

from magiccut.io import read_json, sha256_file, write_json


MAX_CHUNK_BYTES = 450_000_000


def main() -> None:
    benchmark_preparation = read_json("outputs/benchmark_preparation/manifest.json")
    if not benchmark_preparation.get("all_100_valid") or benchmark_preparation.get("failures"):
        raise ValueError("benchmark preparation must be complete and clean before checkpointing")
    files = []
    for uid, row in sorted(benchmark_preparation["archives"].items()):
        path = Path(row["local_path"])
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"Archive digest drift: {uid}")
        files.append((uid, path, path.stat().st_size, row["sha256"]))

    groups = []
    current = []
    current_bytes = 0
    for item in files:
        if current and current_bytes + item[2] > MAX_CHUNK_BYTES:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += item[2]
    if current:
        groups.append(current)

    output = Path("raw_checkpoints")
    output.mkdir(exist_ok=True)
    chunks = []
    for index, group in enumerate(groups, 1):
        path = output / f"magiccut-raw-archives-{index:02d}-of-{len(groups):02d}.tar"
        with tarfile.open(path, "w") as handle:
            for _, source, _, _ in group:
                handle.add(source, arcname=source.as_posix(), recursive=False)
        chunks.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "archives": [
                    {"uid": uid, "path": str(source), "bytes": size, "sha256": digest}
                    for uid, source, size, digest in group
                ],
            }
        )

    support = output / "magiccut-raw-support.tar"
    support_paths = [
        Path("data/raw/checkpoints/checkpoint.pt"),
        Path("data/raw/benchmark/test-bench/val_renders/metadata.json"),
        Path("data/raw/benchmark/test-bench/objaverse_uids.json"),
    ]
    support_paths.extend(sorted(Path("data/raw/benchmark/test-bench/labels").glob("*/*.json")))
    with tarfile.open(support, "w") as handle:
        for source in support_paths:
            handle.add(source, arcname=source.as_posix(), recursive=False)
    support_row = {"path": str(support), "bytes": support.stat().st_size, "sha256": sha256_file(support)}

    manifest = {
        "dataset_revision": benchmark_preparation["dataset_revision"],
        "archive_count": len(files),
        "chunk_count": len(chunks),
        "max_chunk_bytes": MAX_CHUNK_BYTES,
        "chunks": chunks,
        "support": support_row,
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"archive_count": len(files), "chunks": chunks, "support": support_row}, indent=2))


if __name__ == "__main__":
    main()
