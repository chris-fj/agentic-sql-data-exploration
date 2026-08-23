# AGENTS.md

## Commands

Run everything from the repo root via `uv` (Python 3.14, see `.python-version`). `uv sync` installs the project editable, so `backend.*` and top-level `logging_config` imports resolve without setting `PYTHONPATH`.

```bash
uv sync                                     # install all deps (incl. dev group)
uv run uvicorn backend.app:app --reload     # backend on :8000
uv run streamlit run src/frontend/app.py    # frontend on :8501
uv run pytest                               # all tests — no LLM keys, DB, or network needed (~5s)
uv run pytest tests/ -k "name"              # single test
uv run ruff format src/ tests/ && uv run isort src/ tests/   # format (new files only)
uv run bump-my-version bump patch|minor|major          # CAUTION: auto-commits AND creates a signed tag
```

Docker (Makefile targets wrap these with `sudo docker`):

```bash
docker compose up -d --build    # one-shot duckdb-loader builds the DB, then the app starts
docker compose up duckdb-loader # rebuild the DB from data/*.csv only
docker compose down -v          # also wipes the duckdb_data volume
```

## Architecture

- Streamlit frontend (`src/frontend/app.py`) makes **one** POST to FastAPI `/api/sql-agent` (`src/backend/api/sql_agent.py`) with `{"query", "llm": "cloud"|"local"}` and renders the returned report/chart.
- The backend runs a LangGraph `StateGraph` (`src/backend/agents/graph.py`):
  `clarify_intent → route → analyze_schema → generate_sql → validate_sql → execute_sql → (optional) generate_chart → generate_report`; non-data intents go straight to `chat_response`.
- `validate_sql` is a security gate: sqlglot AST static analysis + DuckDB `EXPLAIN` cost analysis. Failures loop back to `generate_sql` (max 3 attempts), then fall through to an error report.
- LLM selection: `get_llm()` in `src/backend/utils/llm.py` — `CLOUD_*` env vars → DeepSeek, `LOCAL_*` → Ollama. `.env` is loaded at import time; missing required vars raise `OSError`. See `.env.example`.
- DuckDB is **embedded** (no server/port): opened read-only from `DB_PATH` (default `/db/sql_agent.db`).
- `main.py` is a leftover stub, not an entrypoint.

## Gotchas

- `DB_PATH` is read at import time into module-level constants (`backend.tools.sql_tool`, `backend.agents.graph`). To point code at a different DB, monkeypatch those module attributes — setting the env var later has no effect (see `temp_duckdb` fixture in `tests/conftest.py`).
- Tests are fully mocked (LLM via `AsyncMock`/`patch`). `conftest.py` has an autouse fixture that strips all `CLOUD_*`/`LOCAL_*` env vars so a local `.env` never leaks in.
- isort is configured to wrap multi-name `from` imports into parentheses, one name per line with a trailing comma (`multi_line_output = 3`, `force_grid_wrap = 2`).
- `scripts/init_db.sh` runs under **dash** in the loader container: no bashisms (no `${var:0:8}`, `[[ ]]`, `set -o pipefail`, `&>`).
- `scripts/Dockerfile.loader` must stay Debian-based (the copied duckdb binary is glibc-linked), and the app Dockerfile must stay `python:3.14-slim` (no musllinux wheels for numpy/pandas/pyarrow on alpine).
- `.dockerignore` excludes `data/` and `.env` — in Docker, CSVs reach the loader only via the `./data:/data:ro` volume mount.
- `logs/` is root-owned (Makefile runs docker with sudo); local writes there can fail with EACCES.

- LangChain/LangGraph docs are available via the MCP servers in `opencode.json` (`docs-langchain`, `reference-langchain`).

## Code style

- All code must be formatted with ruff and isort, following the conventions on `pyproject.toml` (ruff at 88 lines, python 3.14, and isort with parenthesized multi-line imports and trailing commas to reduce diff noise)
- All tests must be written observing the pattern "arrange-act-assert".
- All functions, methods and classes must define a docstring in numpy format. This is enforced by ruff (`D101`, `D102`, `D103`). Example:

  ```python
  def add(a: int, b: int) -> int:
      """Add two integers and return the result.

      Parameters
      ----------
      a : int
          The first operand.
      b : int
          The second operand.

      Returns
      -------
      int
          The sum of `a` and `b`.
      """
  ```

### Formatting policy

- Agents must **never** run `ruff format` on an entire existing file. This reformats code you didn't touch and pollutes diffs.
- For **new files** created from scratch, `uv run ruff format <file>` is acceptable.
- For **existing files**, only format the lines you changed:
  - Inspect first: `uv run ruff format --diff <file>` (dry-run, no modifications)
  - Apply to a range: `uv run ruff format --range <start_line>:<start_col>-<end_line>:<end_col> <file>`
- Use `ruff format --check <file>` to verify a file is already formatted (exit code only, no output).