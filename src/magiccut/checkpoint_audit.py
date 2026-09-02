"""Safe structural audit of the released PyTorch checkpoint."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch

from .io import sha256_file
from .resources import CHECKPOINT_SHA256, CHECKPOINT_SIZE


def _state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "encoder"):
            candidate = payload.get(key)
            if isinstance(candidate, dict) and candidate:
                return candidate
        if payload and all(isinstance(key, str) for key in payload):
            return payload
    raise ValueError("Checkpoint does not contain an identifiable state dictionary")


def audit_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path)
    size = checkpoint.stat().st_size
    digest = sha256_file(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = _state_dict(payload)

    prefixes = Counter(key.split(".", 1)[0] for key in state)
    tensor_shapes = {
        key: list(value.shape)
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    critical_names = [
        "backbone.cls_token",
        "backbone.pos_embed",
        "backbone.patch_embed.proj.weight",
        "head.0.weight",
        "head.0.bias",
        "head.2.weight",
        "head.2.bias",
    ]
    return {
        "path": str(checkpoint),
        "size_bytes": size,
        "size_matches_official": size == CHECKPOINT_SIZE,
        "sha256": digest,
        "sha256_matches_official": digest == CHECKPOINT_SHA256,
        "payload_type": type(payload).__name__,
        "parameter_count": len(tensor_shapes),
        "total_tensor_values": sum(
            value.numel() for value in state.values() if isinstance(value, torch.Tensor)
        ),
        "top_level_prefix_counts": dict(sorted(prefixes.items())),
        "critical_parameter_shapes": {
            name: tensor_shapes[name] for name in critical_names if name in tensor_shapes
        },
        "first_parameter_shapes": dict(list(sorted(tensor_shapes.items()))[:30]),
        "all_parameter_names": sorted(tensor_shapes),
    }
