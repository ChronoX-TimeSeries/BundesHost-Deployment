# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# git is needed for some pip installs that pull from VCS; curl kept for healthchecks/debug
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install all dev deps (prefect, dbt-core, dbt-postgres, mlflow, etc.)
COPY pyproject.toml requirements.txt requirements-dev.txt README.md ./
COPY src/ ./src/
RUN pip install -r requirements-dev.txt && pip install -e .

# Bring in the dbt project so dbt_task can run `dbt run` / `dbt test`
COPY dbt/ ./dbt/

# Install dbt package dependencies (e.g. dbt_utils) into dbt_packages/
RUN cd dbt && dbt deps

# Keep the container alive so flows can be triggered via `docker compose exec`
CMD ["sleep", "infinity"]
