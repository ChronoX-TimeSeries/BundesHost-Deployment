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
def ingest_task(source: str = "api") -> int:
    """Fetch tourism data and UPSERT it into public.tourism_raw."""
    logger = get_run_logger()
    logger.info(f"Ingest started (source={source})")

    df = load_data(source)
    logger.info(f"Loaded {len(df)} rows from {source}")

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