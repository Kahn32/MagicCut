#!/usr/bin/env python3
"""Resume the unusually large final Prompt-34 mesh in exact FP32 chunks."""
from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch

from magiccut.data.renders import safe_extract_tar
from magiccut.inference import collect_render_triplets, encode_triplets
from magiccut.io import read_json, sha256_file, write_json, write_json_atomic
from magiccut.model import load_released_encoder
from magiccut.resources import CHECKPOINT_SHA256, DEDUP_METADATA_PATH


FINAL_UID = "1c1899106e9d41589086dbe752f7d24a"
CHUNK_SIZE = 512  # divisible by the frozen batch size, preserving batch boundaries
TAIL_CHUNK_SIZE = 128  # protect the final large-mesh tail from transport interruptions
BATCH_SIZE = 128


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def valid_part(path: Path, expected_ids: list[int]) -> bool:
    try:
        payload = np.load(path, allow_pickle=False)
        ids, x = payload["ids"], payload["x"]
        return (
            ids.tolist() == expected_ids
            and x.shape == (len(expected_ids), 1152)
            and np.isfinite(x).all()
            and str(payload["checkpoint_sha256"].item()) == CHECKPOINT_SHA256
        )
    except Exception:
        return False


def update_summary(manifest: dict, manifest_path: Path, report_path: Path) -> None:
    complete = [
        row for row in manifest["meshes"].values()
        if row.get("status") in ("encoded", "reused")
    ]
    manifest.update(
        completed_meshes=len(complete),
        representatives_cached=sum(row.get("representatives", 0) for row in complete),
        all_100_complete=len(complete) == 100 and not manifest["failures"],
        cache_bytes=sum(Path(row["cache_path"]).stat().st_size for row in complete),
    )
    write_json_atomic(manifest_path, manifest)
    write_json(report_path, {key: value for key, value in manifest.items() if key != "meshes"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chunks", type=int, default=1)
    args = parser.parse_args()

    output = Path("reports/generated/embeddings")
    manifest_path = output / "manifest.json"
    report_path = output / "report.json"
    manifest = read_json(manifest_path)
    p33 = read_json("reports/generated/benchmark_preparation/manifest.json")
    metadata = read_json(Path("data/raw/benchmark") / DEDUP_METADATA_PATH)
    expected_ids = sorted(map(int, metadata[FINAL_UID]["unique_ids"]))
    if len(expected_ids) != 4535:
        raise ValueError(f"Unexpected final-mesh size: {len(expected_ids)}")

    checkpoint = Path("data/raw/checkpoints/checkpoint.pt")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint integrity failure")
    archive_info = p33["archives"][FINAL_UID]
    archive = Path(archive_info["local_path"])
    if sha256_file(archive) != archive_info["sha256"]:
        raise ValueError("Archive SHA drift")

    extract_root = Path("data/work") / f"final-{FINAL_UID}-{archive_info['sha256'][:12]}"
    marker = extract_root / ".archive_sha256"
    if not marker.exists():
        extract_root.mkdir(parents=True, exist_ok=True)
        paths = safe_extract_tar(archive, extract_root)
        marker.write_text(archive_info["sha256"] + "\n")
    else:
        if marker.read_text().strip() != archive_info["sha256"]:
            raise ValueError("Persistent extraction marker mismatch")
        paths = sorted(extract_root.rglob("*.png"))

    triplets = collect_render_triplets(paths)
    if len(triplets) != len(expected_ids):
        raise ValueError("Triplet count mismatch")
    if [item.part_id for item in triplets] != expected_ids:
        raise ValueError("Triplet order/ID mismatch")

    parts_root = Path("data/cache/final_mesh_parts")
    chunk_specs = []
    tail_start = (len(triplets) // CHUNK_SIZE) * CHUNK_SIZE
    ranges = [
        (start, start + CHUNK_SIZE)
        for start in range(0, tail_start, CHUNK_SIZE)
    ]
    ranges.extend(
        (start, min(start + TAIL_CHUNK_SIZE, len(triplets)))
        for start in range(tail_start, len(triplets), TAIL_CHUNK_SIZE)
    )
    for start, end in ranges:
        path = parts_root / f"{FINAL_UID}-{start:05d}-{end:05d}.npz"
        chunk_specs.append((start, end, path))

    row = manifest["meshes"].setdefault(FINAL_UID, {})
    row.update(
        status="running_chunked",
        representatives_expected=len(expected_ids),
        chunk_size=CHUNK_SIZE,
        batch_size_used=BATCH_SIZE,
        chunks_total=len(chunk_specs),
        source="full_fp32_chunked_extraction",
        archive_sha256=archive_info["sha256"],
    )

    torch.set_num_threads(min(9, os.cpu_count() or 1))
    model, model_audit = load_released_encoder(checkpoint)
    model.eval()
    manifest["model"] = model_audit
    completed_this_run = 0
    for index, (start, end, path) in enumerate(chunk_specs, 1):
        ids_for_part = expected_ids[start:end]
        if valid_part(path, ids_for_part):
            print(f"chunk {index}/{len(chunk_specs)} cached ({end}/{len(expected_ids)})", flush=True)
            continue
        if completed_this_run >= args.max_chunks:
            continue
        began = time.perf_counter()
        ids, x, _ = encode_triplets(model, triplets[start:end], batch_size=BATCH_SIZE)
        if ids.tolist() != ids_for_part or x.shape != (end - start, 1152):
            raise ValueError(f"Chunk {index} output mismatch")
        atomic_npz(
            path,
            ids=ids,
            x=x,
            checkpoint_sha256=np.asarray(CHECKPOINT_SHA256),
        )
        if not valid_part(path, ids_for_part):
            raise ValueError(f"Chunk {index} failed post-write validation")
        row.setdefault("chunk_elapsed_seconds", {})[str(index)] = time.perf_counter() - began
        completed_this_run += 1
        print(f"chunk {index}/{len(chunk_specs)} encoded ({end}/{len(expected_ids)})", flush=True)

    valid_specs = [
        (start, end, path) for start, end, path in chunk_specs
        if valid_part(path, expected_ids[start:end])
    ]
    row.update(
        chunks_completed=len(valid_specs),
        representatives_completed=sum(end - start for start, end, _ in valid_specs),
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )

    if len(valid_specs) == len(chunk_specs):
        ids_arrays, x_arrays = [], []
        for _, _, path in chunk_specs:
            payload = np.load(path, allow_pickle=False)
            ids_arrays.append(payload["ids"])
            x_arrays.append(payload["x"])
        ids = np.concatenate(ids_arrays)
        x = np.concatenate(x_arrays)
        if ids.tolist() != expected_ids or x.shape != (len(expected_ids), 1152):
            raise ValueError("Merged final cache mismatch")
        cache = Path("data/cache/full_embeddings") / (
            f"{FINAL_UID}-{archive_info['sha256'][:12]}-{CHECKPOINT_SHA256[:12]}.npz"
        )
        atomic_npz(
            cache,
            ids=ids,
            x=x,
            checkpoint_sha256=np.asarray(CHECKPOINT_SHA256),
        )
        if not valid_part(cache, expected_ids):
            raise ValueError("Final cache failed validation")
        row.update(
            status="encoded",
            cache_path=str(cache),
            representatives=len(expected_ids),
            x_shape=[len(expected_ids), 1152],
            cache_sha256=sha256_file(cache),
            elapsed_seconds=sum(row["chunk_elapsed_seconds"].values()),
        )
        if FINAL_UID in manifest["failures"]:
            manifest["failures"].remove(FINAL_UID)

    update_summary(manifest, manifest_path, report_path)
    print(json.dumps({
        "completed_meshes": manifest["completed_meshes"],
        "all_100_complete": manifest["all_100_complete"],
        "final_mesh_status": row["status"],
        "chunks_completed": row["chunks_completed"],
        "chunks_total": row["chunks_total"],
        "representatives_completed": row["representatives_completed"],
        "representatives_expected": row["representatives_expected"],
        "failures": manifest["failures"],
    }, indent=2))


if __name__ == "__main__":
    main()
