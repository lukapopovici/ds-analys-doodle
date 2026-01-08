## Docker / Container usage 

This project is a Streamlit app. The provided `Dockerfile` supports building dependencies from one of these lockfile formats (in preference order):

- `poetry.lock` (uses `poetry export`)
- `Pipfile.lock` (uses `pipenv lock`)
- `uv.lock` (treated like a `requirements.txt` file)
- `requirements.txt`
- If no lockfile exists, the image falls back to installing the package from `pyproject.toml`.

Quick start (build and run with Docker):

- Build: `docker build -t ds-analys-doodle .`
- Run: `docker run -p 8501:8501 ds-analys-doodle`

Or with Docker Compose (recommended for local dev):

- Start: `docker compose up --build`
- Visit: `http://localhost:8501`

Notes:
- The `Dockerfile` aims for reproducible installs by preferring lockfiles when they exist.
- If you're using a dependency manager not listed above, let me know and I can add explicit support.
