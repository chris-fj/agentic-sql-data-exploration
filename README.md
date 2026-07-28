# sql-agent

A Streamlit + FastAPI application. The frontend provides a chat-style text interface; the backend serves API endpoints.

## Quick start

```bash
uv sync
uv run uvicorn backend.app:app --reload &   # backend on :8000
uv run streamlit run src/frontend/app.py    # frontend on :8501
```

## The tool
Takes user queries in natural language, understands them and converts them into SQL that is reviewed and produces an answer. The query produced query is reviewed to determine malicious or unintentedly degrading queries

![prompt, summary](./imgs/img1.png)
![table](./imgs/img2.png)
![plot](./imgs/img3.png)

## Version bumping

This project uses [bump-my-version](https://github.com/callowayproject/bump-my-version).

```bash
uv run bump-my-version bump patch   # 0.1.0 → 0.1.1
uv run bump-my-version bump minor   # 0.1.0 → 0.2.0
uv run bump-my-version bump major   # 0.1.0 → 1.0.0
```

Each bump updates `pyproject.toml`, commits the change, and creates a git tag.
