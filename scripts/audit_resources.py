#!/usr/bin/env python3
"""Download and audit the released Material Magic Wand checkpoint/resources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from magiccut.checkpoint_audit import audit_checkpoint
from magiccut.io import write_json
from magiccut.resources import CHECKPOINT_FILENAME, MODEL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-checkpoint", action="store_true")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    api = HfApi()
    model_info = api.model_info(MODEL.repo_id, revision=MODEL.revision)
    report: dict[str, object] = {
        "model_repo": MODEL.repo_id,
        "model_revision": model_info.sha,
        "model_files": sorted(sibling.rfilename for sibling in model_info.siblings),
    }

    if args.download_checkpoint:
        target = args.raw_root / "checkpoints"
        downloaded = hf_hub_download(
            repo_id=MODEL.repo_id,
            filename=CHECKPOINT_FILENAME,
            revision=MODEL.revision,
            local_dir=target,
        )
        report["checkpoint"] = audit_checkpoint(downloaded)

    write_json("outputs/resource_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
