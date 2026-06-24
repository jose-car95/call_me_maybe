.PHONY: install run trace debug clean fclean lint lint-strict test

install:
	uv sync

run:
	uv run python -m src

trace:
	uv run python -m src --trace

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +

fclean: clean
	$(RM) -r data/output .venv

lint:
	uv run flake8 .
	uv run pydocstyle --convention=google src
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run pydocstyle --convention=google src
	uv run mypy . --strict

test:
	uv run pytest
