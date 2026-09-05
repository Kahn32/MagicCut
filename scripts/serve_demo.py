#!/usr/bin/env python3
"""Serve the local interactive demo MagicCut demo."""

from __future__ import annotations

import io
import tarfile
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from magiccut.data.renders import parse_render_name
from magiccut.demo import MagicCutDemo
from magiccut.io import read_json


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "demo"
app = FastAPI(title="MagicCut demo", version="0.1.0")
engine: MagicCutDemo | None = None


class StartRequest(BaseModel):
    uid: str
    query_part: int


class FeedbackRequest(BaseModel):
    session_id: str
    part_id: int
    positive: bool


class ResetRequest(BaseModel):
    session_id: str


def get_engine() -> MagicCutDemo:
    global engine
    if engine is None:
        engine = MagicCutDemo(ROOT)
    return engine


@lru_cache(maxsize=32)
def render_member(uid: str, part_id: int) -> tuple[Path, str]:
    manifest = read_json(ROOT / "outputs/benchmark_preparation/manifest.json")
    if uid not in manifest["archives"]:
        raise KeyError(uid)
    archive = Path(manifest["archives"][uid]["local_path"])
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            if not member.isfile():
                continue
            try:
                parsed = parse_render_name(member.name)
            except ValueError:
                continue
            if parsed.part_id == part_id and parsed.size == "medium":
                return archive, member.name
    raise KeyError((uid, part_id))


def as_http_error(exc: Exception) -> HTTPException:
    status = 404 if isinstance(exc, KeyError) else 400
    return HTTPException(status_code=status, detail=str(exc))


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/meshes")
def meshes():
    try:
        return get_engine().meshes()
    except Exception as exc:
        raise as_http_error(exc) from exc


@app.post("/api/start")
def start(request: StartRequest):
    try:
        return get_engine().start(request.uid, request.query_part)
    except Exception as exc:
        raise as_http_error(exc) from exc


@app.post("/api/feedback")
def feedback(request: FeedbackRequest):
    try:
        return get_engine().feedback(request.session_id, request.part_id, request.positive)
    except Exception as exc:
        raise as_http_error(exc) from exc


@app.post("/api/reset")
def reset(request: ResetRequest):
    try:
        return get_engine().reset(request.session_id)
    except Exception as exc:
        raise as_http_error(exc) from exc


@app.get("/api/render/{uid}/{part_id}")
def render(uid: str, part_id: int):
    try:
        archive, member_name = render_member(uid, part_id)
        with tarfile.open(archive, "r:gz") as handle:
            source = handle.extractfile(member_name)
            if source is None:
                raise KeyError((uid, part_id))
            payload = io.BytesIO(source.read()).getvalue()
        return Response(payload, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})
    except Exception as exc:
        raise as_http_error(exc) from exc


app.mount("/static", StaticFiles(directory=WEB), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
