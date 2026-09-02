"""Small, deterministic file utilities."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_json_atomic(path: str | Path, value: Any) -> None:
    destination=Path(path);destination.parent.mkdir(parents=True,exist_ok=True)
    handle=tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=destination.parent,prefix=destination.name+".",delete=False)
    try:
        with handle:
            json.dump(value,handle,indent=2,sort_keys=True);handle.write("\n");handle.flush();os.fsync(handle.fileno())
        os.replace(handle.name,destination)
    except Exception:
        try:os.unlink(handle.name)
        except FileNotFoundError:pass
        raise


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
