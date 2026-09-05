#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "$0")/.." && pwd)
clean_root=$(mktemp -d "${TMPDIR:-/tmp}/magiccut-clean-XXXXXX")
trap 'rm -rf "$clean_root"' EXIT

tar -C "$project_root" \
  --exclude=.venv --exclude=data/raw --exclude=data/cache --exclude=data/work \
  --exclude=raw_checkpoints --exclude=.pytest_cache --exclude='*/__pycache__' \
  --exclude=outputs/ \
  --exclude=reports/ \
  -cf - . | tar -C "$clean_root" -xf -

cd "$clean_root"
uv sync --frozen --extra dev
uv run pytest -q
uv run python scripts/audit_git_payload.py
