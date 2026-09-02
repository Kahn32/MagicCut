#!/usr/bin/env python3
"""Reject accidental large files and private workspace identifiers."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IGNORED_ROOTS = (
    ROOT / ".venv",
    ROOT / "data/raw",
    ROOT / "data/cache",
    ROOT / "data/work",
    ROOT / "raw_checkpoints",
    ROOT / "reports/generated/locked_evaluation/progress.json",
    ROOT / "reports/generated/locked_evaluation/raw_results.json",
)
LIMIT = 10 * 1024 * 1024
PRIVATE_MARKERS = (
    b"lib" + b"file_",
    b"oai-" + b"library://",
    b"/work" + b"space/scratch/",
    b"joining " + b"TISL strategy",
)


def ignored(path: Path) -> bool:
    return (
        any(part in {"__pycache__", ".pytest_cache"} for part in path.parts)
        or any(root == path or root in path.parents for root in IGNORED_ROOTS)
        or (
        path.parent == ROOT / "reports/generated/locked_evaluation"
        and path.name.startswith("progress.json.")
        )
    )


def main() -> None:
    offenders = []
    private_hits = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ignored(path) or ".git" in path.parts:
            continue
        if path.stat().st_size > LIMIT:
            offenders.append((str(path.relative_to(ROOT)), path.stat().st_size))
            continue
        payload = path.read_bytes()
        for marker in PRIVATE_MARKERS:
            if marker in payload:
                private_hits.append((str(path.relative_to(ROOT)), marker.decode("ascii")))
    if offenders:
        raise SystemExit(f"Large non-data files found: {offenders}")
    if private_hits:
        raise SystemExit(f"Private workspace identifiers found: {private_hits}")
    print(f"repository payload audit passed (limit={LIMIT} bytes, no private identifiers)")


if __name__ == "__main__":
    main()
