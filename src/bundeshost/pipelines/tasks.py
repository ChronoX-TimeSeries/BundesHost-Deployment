"""Prefect tasks used by the BundesHost retrain flow.

Each task wraps an existing piece of the modeling/data pipeline so it gets:
- Run history in Prefect Cloud
- Retries on transient failures
- Logging tied to the parent flow

Tasks here are kept thin — they call into bundeshost.* modules and don't
duplicate any business logic.
"""

import subprocess
from pathlib import Path

from prefect import get_run_logger, task

from bundeshost.config import BASE_DIR
from bundeshost.data.ingest import get_engine, load_data, upsert_tourism_raw

# ---------------------------------------------------------------------------
# Ingest (re-export of what ingest_flow already uses, kept here for the
# quarterly flow to import everything from one place).
# ---------------------------------------------------------------------------


@task(retries=2, retry_delay_seconds=10, name="ingest")
def ingest_task(source: str = "api", df=None) -> int:
    """Fetch tourism data and UPSERT it into public.tourism_raw.

    If `df` is provided (e.g. from check_for_new_data_task), skip fetching
    and use it directly. Otherwise fetch from `source` ('api' or 'csv').
    """
    logger = get_run_logger()

    if df is None:
        logger.info(f"Ingest started (source={source})")
        df = load_data(source)
        logger.info(f"Loaded {len(df)} rows from {source}")
    else:
        logger.info(f"Ingest started (using pre-fetched DataFrame, {len(df)} rows)")

    engine = get_engine()
    total = upsert_tourism_raw(df, engine)

    logger.info(f"UPSERT complete; tourism_raw now has {total} rows")
    return total


# ---------------------------------------------------------------------------
# dbt
# ---------------------------------------------------------------------------


def _run_dbt(command: list[str]) -> str:
    """Run a dbt command from inside dbt/. Raises if it exits non-zero."""
    dbt_dir = Path(BASE_DIR) / "dbt"
    result = subprocess.run(
        command,
        cwd=dbt_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(command)} failed (exit {result.returncode})\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout


@task(retries=1, retry_delay_seconds=15, name="dbt-run-and-test")
def dbt_task(target: str = "dev") -> None:
    """Rebuild the dbt mart and run its tests.

    Runs `dbt run --target <target>` followed by `dbt test --target <target>`.
    Both must succeed for the task to pass.
    """
    logger = get_run_logger()

    logger.info(f"dbt run --target {target}")
    run_out = _run_dbt(["dbt", "run", "--target", target])
    logger.info(run_out.strip().splitlines()[-1] if run_out.strip() else "dbt run finished")

    logger.info(f"dbt test --target {target}")
    test_out = _run_dbt(["dbt", "test", "--target", target])
    logger.info(test_out.strip().splitlines()[-1] if test_out.strip() else "dbt test finished")


# ---------------------------------------------------------------------------
# Evaluate & Train
#
# These wrap the existing modeling entrypoints. evaluate_all_states picks the
# best variant per state and writes best_models.json; retrain_all_states then
# reads that file, fits on the full data, registers in MLflow, and auto-promotes
# if the new MAPE beats the current Production MAPE.
# ---------------------------------------------------------------------------


@task(retries=0, name="evaluate-all-states")
def evaluate_task() -> dict:
    """Compare SARIMA vs SARIMAX per state on the 12-month holdout, log to MLflow.

    Writes models/best_models.json. Returns the results dict.
    """
    from bundeshost.modeling.evaluate import evaluate_all_states

    logger = get_run_logger()
    logger.info("Starting evaluate_all_states (SARIMA vs SARIMAX, 12-month holdout)")
    results = evaluate_all_states()
    logger.info(f"Evaluation finished for {len(results)} states")
    return results


@task(retries=0, name="retrain-all-states")
def train_task() -> None:
    """Refit the winning variant per state on the full data, register + auto-promote."""
    from bundeshost.modeling.train import retrain_all_states

    logger = get_run_logger()
    logger.info("Starting retrain_all_states (full-data refit, MLflow register + promote)")
    retrain_all_states()
    logger.info("Retrain finished")


# ---------------------------------------------------------------------------
# Conditional check: does Destatis have data newer than what we already have?
# ---------------------------------------------------------------------------


@task(retries=2, retry_delay_seconds=10, name="check-for-new-data")
def check_for_new_data_task() -> dict:
    """Fetch from Destatis and compare with the latest date already in Postgres.

    Returns a dict with:
        has_new_data (bool): True if the API has months newer than the DB
        latest_in_db (date or None): max(date) in public.tourism_raw, or None if empty
        latest_in_api (date): max(date) from the API response
        df (DataFrame): the full API response, so ingest_task doesn't refetch

    Reused by quarterly_retrain to decide whether to continue or early-exit.
    """
    from sqlalchemy import text

    from bundeshost.data.destatis_client import fetch_from_api

    logger = get_run_logger()
    logger.info("Fetching latest snapshot from Destatis API")
    df = fetch_from_api()
    latest_in_api = df["date"].max()
    logger.info(f"API latest date: {latest_in_api.date()}")

    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT MAX(date) FROM public.tourism_raw")).scalar()
    latest_in_db = result  # may be None if the table is empty
    logger.info(f"DB latest date:  {latest_in_db}")

    has_new_data = (latest_in_db is None) or (latest_in_api.date() > latest_in_db)
    logger.info(f"has_new_data = {has_new_data}")

    return {
        "has_new_data": has_new_data,
        "latest_in_db": latest_in_db,
        "latest_in_api": latest_in_api.date(),
        "df": df,
    }


# ---------------------------------------------------------------------------
# Cache invalidation: tell the API to drop its in-memory model cache after
# new models are promoted to Production.
# ---------------------------------------------------------------------------


@task(retries=3, retry_delay_seconds=5, name="invalidate-api-cache")
def invalidate_api_cache_task() -> dict:
    """POST to the API's /admin/clear-cache endpoint.

    Uses API_BASE_URL from env (default http://localhost:8000).
    Returns the JSON response. Soft-fails: if the API is unreachable, logs
    a warning rather than failing the whole flow — the cache will clear
    next time the API restarts anyway.
    """
    import os

    import httpx

    logger = get_run_logger()
    base = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
    url = f"{base}/admin/clear-cache"

    try:
        response = httpx.post(url, timeout=10.0)
        response.raise_for_status()
        result = response.json()
        logger.info(f"API cache cleared: {result}")
        return result
    except httpx.HTTPError as e:
        logger.warning(
            f"Could not reach API at {url} ({e!r}). " f"Cache will clear on next API restart."
        )
        return {"status": "unreachable", "error": str(e)}


# ---------------------------------------------------------------------------
# Drift monitoring: compare the latest 12 months of arrivals against the
# 24 months immediately preceding them. Soft-fail — drift detection is a
# passive observation, not a decision input. If anything goes wrong, log
# a warning and continue; the rest of the flow doesn't depend on it.
# ---------------------------------------------------------------------------


@task(retries=0, name="drift-report")
def drift_report_task() -> dict:
    """Generate a per-state data-drift report and log it to MLflow.

    Reads the latest mart data via pipeline.get_tourism_data(), then for each
    state slices a reference window (24 months) and a current window (12
    months) and runs Evidently's DataDriftPreset on arrivals. Logs to MLflow
    under experiment 'bundeshost-monitoring': a per_state_drift.json artifact
    (the results dict), an n_drifted_states metric, and the combined HTML
    report.

    Returns a dict with status, run_id, n_drifted_states, and the list of
    drifted_states. Never raises — drift reporting must not fail the retrain
    flow.
    """
    logger = get_run_logger()

    try:
        import tempfile
        from pathlib import Path

        import mlflow

        from bundeshost.data.pipeline import get_tourism_data
        from bundeshost.monitoring.drift import generate_per_state_drift

        logger.info("Loading tourism data for drift report")
        df = get_tourism_data()

        logger.info("Generating per-state Evidently data-drift report (arrivals)")
        results, html = generate_per_state_drift(df)

        drifted_states = [s for s, info in results.items() if info.get("drift")]
        logger.info(
            f"Drift detected in {len(drifted_states)}/{len(results)} states: "
            f"{drifted_states or 'none'}"
        )

        mlflow.set_experiment("bundeshost-monitoring")
        with mlflow.start_run(run_name="drift-report") as run:
            mlflow.log_dict(results, "per_state_drift.json")
            mlflow.log_metric("n_drifted_states", len(drifted_states))
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "drift_report.html"
                path.write_text(html)
                mlflow.log_artifact(str(path))
            logger.info(f"Drift report logged to MLflow run {run.info.run_id}")
            return {
                "status": "ok",
                "run_id": run.info.run_id,
                "n_drifted_states": len(drifted_states),
                "drifted_states": drifted_states,
            }

    except Exception as e:
        logger.warning(f"Drift report failed ({e!r}); continuing flow.")
        return {"status": "failed", "error": str(e)}
