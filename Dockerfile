# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# Install system deps required for some Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata and any lockfiles early for better Docker cache layering
COPY pyproject.toml .
COPY poetry.lock* ./
COPY Pipfile* ./
COPY Pipfile.lock* ./
COPY uv.lock* ./
COPY requirements.txt* ./

# Upgrade pip and install dependencies using whichever lockfile exists
RUN python -m pip install --upgrade pip setuptools wheel

# The install logic below prefers a lockfile when available:
#  - poetry.lock -> use poetry to export requirements and pip install them
#  - Pipfile.lock -> use pipenv to export requirements and pip install them
#  - uv.lock -> treated as a requirements-like file and pip installed
#  - requirements.txt -> pip install it
#  - fallback -> install the package from source (pyproject)
RUN if [ -f poetry.lock ]; then \
        pip install "poetry>=1.2.0" && \
        poetry export -f requirements.txt --without-hashes -o /tmp/requirements.txt && \
        pip install -r /tmp/requirements.txt; \
    elif [ -f Pipfile.lock ]; then \
        pip install pipenv && \
        pipenv lock -r > /tmp/requirements.txt && \
        pip install -r /tmp/requirements.txt; \
    elif [ -f uv.lock ]; then \
        pip install -r uv.lock; \
    elif [ -f requirements.txt ]; then \
        pip install -r requirements.txt; \
    else \
        pip install .; \
    fi

# Copy application source
COPY src/ ./src
COPY README.md ./

# Expose Streamlit default port
EXPOSE 8501

# Default command: run Streamlit app
CMD ["streamlit", "run", "src/main.py", "--server.port", "8501", "--server.headless", "true"]
