"""Canonical identifiers and integrity information for official resources."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HubResource:
    repo_id: str
    repo_type: str
    revision: str | None = None


MODEL = HubResource(
    repo_id="umangijain/material-magic-wand",
    repo_type="model",
    revision="cfac3017592ab6925f3efe46ab90e53027c3ba45",
)
DATASET = HubResource(
    repo_id="umangijain/material-magic-wand",
    repo_type="dataset",
)

CHECKPOINT_FILENAME = "checkpoint.pt"
CHECKPOINT_SIZE = 90_249_468
CHECKPOINT_SHA256 = "d86f28e222584d25be3d0d5017de60a88bf45b574d2bf5c73e12010ecfeefe97"

UIDS_PATH = "test-bench/objaverse_uids.json"
LABELS_PREFIX = "test-bench/labels/"
DEDUP_METADATA_PATH = "test-bench/val_renders/metadata.json"
FULL_METADATA_PATH = "test-bench/val_renders_full/metadata.json"
DEDUP_IMAGES_PREFIX = "test-bench/val_renders/images/"
