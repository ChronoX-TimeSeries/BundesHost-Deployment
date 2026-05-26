"""Deploy the quarterly_retrain flow to Prefect Cloud with a quarterly schedule.

This registers the flow as a deployment with a cron schedule (quarterly).
For Hobby tier + managed work pools, code is pulled from GitHub at run time.
Secrets are embedded into the deployment from .env so the managed worker
can connect to Supabase, DagsHub, and Destatis without a localhost.

Usage:
    source .env (or: set -a && source .env && set +a)
    python -m bundeshost.pipelines.deploy
"""

import os

from dotenv import load_dotenv
from prefect import flow

load_dotenv()

QUARTERLY_CRON = "0 2 1 */3 *"
WORK_POOL_NAME = "bundeshost-pool"
GITHUB_REPO = "https://github.com/ChronoX-TimeSeries/BundesHost-Deployment.git"
FLOW_ENTRYPOINT = "src/bundeshost/pipelines/quarterly_retrain.py:quarterly_retrain_flow"


def _env(*names: str) -> str:
    """Return the first non-empty env var among names, or raise."""
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    raise RuntimeError(f"None of these env vars are set: {names}")


def main():
    # Build the env dict that managed workers will see at runtime.
    # Maps SUPABASE_* (which lives in .env for cloud) onto the POSTGRES_*
    # names the flow code actually reads.
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
        # API_BASE_URL points the cache-invalidation task at the public API.
        "API_BASE_URL": os.environ.get(
            "API_BASE_URL_PUBLIC", "https://bundeshost-api.fly.dev"
        ),
    }

    flow.from_source(
        source=GITHUB_REPO,
        entrypoint=FLOW_ENTRYPOINT,
    ).deploy(
        name="quarterly-retrain-prod",
        work_pool_name=WORK_POOL_NAME,
        cron=QUARTERLY_CRON,
        description=(
            "Quarterly retrain pipeline: check Destatis for new data, "
            "if found run ingest -> dbt -> evaluate -> train, "
            "auto-promote in MLflow if MAPE improves. "
            "Phase F+: Supabase-backed, Fly.io API target."
        ),
        tags=["phase-2f-plus", "production"],
        job_variables={"env": runtime_env},
    )


if __name__ == "__main__":
    main()
