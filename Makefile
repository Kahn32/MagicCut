.PHONY: install test audit-data extract evaluate statistics figures demo final-audit clean-check

install:
	uv sync --frozen --extra dev

test:
	uv run pytest -q

audit-data:
	uv run python scripts/prepare_benchmark.py

extract:
	uv run python scripts/extract_embeddings.py

evaluate:
	uv run python scripts/evaluate_locked.py

statistics:
	uv run python scripts/analyze_statistics.py

figures:
	uv run python scripts/generate_figures.py

demo:
	uv run python scripts/serve_demo.py

final-audit:
	uv run python scripts/audit_project.py

clean-check:
	uv run python scripts/audit_git_payload.py
	bash scripts/test_clean_environment.sh
