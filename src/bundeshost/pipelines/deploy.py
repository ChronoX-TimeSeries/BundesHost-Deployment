"""Deploy the quarterly_retrain flow to Prefect Cloud with a quarterly schedule.

Phase G: uses a prebuilt Docker image from ghcr.io that already has bundeshost
installed. The managed worker pulls the image, no git clone or pip install needed.

Usage:
    source .env (or: set -a && source .env && set +a)
    python -m bundeshost.pipelines.deploy
"""

import os

from dotenv import load_dotenv

from bundeshost.pipelines.quarterly_retrain import quarterly_retrain_flow

load_dotenv()

QUARTERLY_CRON = "0 2 1 */3 *"
WORK_POOL_NAME = "bundeshost-pool"
IMAGE = "ghcr.io/chronox-timeseries/bundeshost-prefect:latest"


def _env(*names: str) -> str:
    """Return the first non-empty env var among names, or raise."""
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    raise RuntimeError(f"None of these env vars are set: {names}")


def main():
    runtime_env = {
        "POSTGRES_HOST": _env("SUPABASE_POSTGRES_HOST"),
        "POSTGRES_PORT": _env("SUPABASE_POSTGRES_PORT"),
        "POSTGRES_DB": _env("SUPABASE_POSTGRES_DB"),
        "POSTGRES_USER": _env("SUPABASE_POSTGRES_USER"),
        "POSTGRES_PASSWORD": _env("SUPABASE_POSTGRES_PASSWORD"),
        "DBT_MART_SCHEMA": os.environ.get("DBT_MART_SCHEMA", "dbt_dev_mart"),
        "MLFLOW_TRACKING_URI": _env("MLFLOW_TRACKING_URI"),
        "MLFLOW_TRACKING_USERNAME": _env("MLFLOW_TRACKING_USERNAME"),
        "MLFLOW_TRACKING_PASSWORD": _env("MLFLOW_TRACKING_PASSWORD"),
        "DESTATIS_API_BASE_URL": _env("DESTATIS_API_BASE_URL"),
        "DESTATIS_API_TOKEN": _env("DESTATIS_API_TOKEN"),
        "API_BASE_URL": os.environ.get("API_BASE_URL_PUBLIC", "https://bundeshost-api.fly.dev"),
    }

    quarterly_retrain_flow.deploy(
        name="quarterly-retrain-prod",
        work_pool_name=WORK_POOL_NAME,
        image=IMAGE,
        build=False,
        push=False,
        cron=QUARTERLY_CRON,
        description=(
            "Quarterly retrain pipeline: check Destatis for new data, "
            "if found run ingest -> dbt -> evaluate -> train, "
            "auto-promote in MLflow if MAPE improves. "
            "Phase G: ghcr.io image-based, no git clone."
        ),
        tags=["phase-2g", "production"],
        job_variables={"env": runtime_env},
    )


if __name__ == "__main__":
    main()
