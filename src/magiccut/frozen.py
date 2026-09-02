"""Configuration frozen on validation data before the full evaluation."""

from __future__ import annotations

from magiccut.resources import CHECKPOINT_SHA256


DATASET_REVISION = "52b0489beef04a81453944dba4d3c098fbaffe60"
VAL_UIDS = (
    "ffc0e9978ea342739e7abd6abcb1a437",
    "5b1ea87674ca4c5583d55129db284aad",
    "c2ca0c6777a94f53bf38a3cb440e619b",
    "5b01e3e82b8742709a2380056ce1599b",
    "02c42cbb51c24963b9c99d5762547bf2",
    "49d7158c62864ba68cc787ca39a4dfb3",
)
DIRECT_THRESHOLD = 1575.9833984375
ADAPTIVE_THRESHOLD = 0.9792989870
PROBABILITY_THRESHOLD = 0.3222533223833185
GRAPH = {"k": 50, "pairwise_lambda": 0.25, "sigma_factor": 0.5, "edge_feature": "part"}
INTERACTION = {
    "clarification_policy": "influence_aware",
    "uncertainty_method": "calibration_entropy",
    "weights": {"uncertainty": 0.5, "degree": 0.0, "diversity": 0.5},
    "click_budget": 3,
}


def locked_config() -> dict:
    return {
        "dataset_revision": DATASET_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "validation_uids": list(VAL_UIDS),
        "direct_threshold": DIRECT_THRESHOLD,
        "adaptive_threshold": ADAPTIVE_THRESHOLD,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "graph": GRAPH,
        "interaction": INTERACTION,
        "bootstrap_mesh_samples": 2000,
        "bootstrap_seed": 0,
        "test_time_tuning": False,
    }
