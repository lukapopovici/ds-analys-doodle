# ds-analys-doodle

A small Streamlit-based data analysis playground.

## Requirements

- Python 3.10 or newer
- Optional: Docker and Docker Compose for containerized runs
- Dependency management: this project uses a lockfile `uv.lock`. 

## Local development (recommended)

1. Clone the repository:

```bash
git clone <repo-url>
cd ds-analys-doodle
```

2. Preferred: use `uv` to manage the environment and install dependencies from `uv.lock`:

- If you use `uv`, follow your normal `uv` workflow to create/activate the virtual environment and install packages from `uv.lock` (see `uv` documentation for exact commands).

3. Alternative: create a standard Python venv and install from the lockfile (fallback):

4. Run the app with Streamlit:

```bash
cd src
streamlit run src/main.py --server.port 8501 --server.headless true
```


## Docker (optional)

Build and run with Docker:

```bash
docker build -t ds-analys-doodle .
docker run -p 8501:8501 ds-analys-doodle
```

Or use Docker Compose for local development (the compose file binds the project directory so changes are visible inside the container):

```bash
docker compose up --build
```

## Terminal logging (runtime)

The app includes a singleton in `utils/terminal.py` for optional runtime logging. To enable logging in the running app, open the app and use the sidebar toggle labeled "Enable terminal logging". From anywhere in the code you can log messages like this:

```python
from utils.terminal import terminal
terminal.log("Some message", level="INFO")
```

When enabled, messages are stored in memory; you can clear the stored history using the "Clear terminal history" button in the sidebar.

## Updating dependencies and the image

- If you add or update dependencies, update the appropriate lockfile for your package manager and commit it (for example `uv.lock` or `poetry.lock`).
- Rebuild the Docker image after changes with `docker compose up --build` or `docker build ...`.

## Notes and tips

- The app expects data to be loaded via the Upload page (CSV input) or from session state.
- For quick iteration, use `pip install -e .` and run Streamlit locally; Docker Compose with a bind mount also supports fast reloads.

If you want, I can add a short troubleshooting section or a development checklist next. Let me know which you'd prefer.