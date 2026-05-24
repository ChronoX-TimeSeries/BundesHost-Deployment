"""Quarterly retrain flow.

Wires the four tasks together in dependency order:
    ingest -> dbt -> evaluate -> train

What still needs to be added in later steps:
- Step 6: conditional check ("only proceed if Destatis has new data")
- Step 7: cron schedule
- Step 8: notification on completion

Run locally:
    python -m bundeshost.pipelines.quarterly_retrain
    python -m bundeshost.pipelines.quarterly_retrain --source csv
"""

import argparse

from prefect import flow, get_run_logger

from bundeshost.pipelines.tasks import dbt_task, evaluate_task, ingest_task, train_task


@flow(name="quarterly-retrain")
def quarterly_retrain_flow(source: str = "api", dbt_target: str = "dev") -> None:
    """End-to-end retrain pipeline.

    1. Ingest fresh data into public.tourism_raw (idempotent UPSERT)
    2. Rebuild the dbt mart and run its tests
    3. Evaluate SARIMA vs SARIMAX on the 12-month holdout, pick winners
    4. Refit winners on the full data, register in MLflow, auto-promote if MAPE improves
    """
    logger = get_run_logger()
    logger.info(f"Starting quarterly-retrain (source={source}, dbt_target={dbt_target})")

    # Step 1: ingest
    row_count = ingest_task(source=source)

    # Step 2: dbt (waits for ingest via wait_for)
    dbt_task(target=dbt_target, wait_for=[row_count])

    # Step 3: evaluate (waits for dbt)
    evaluate_results = evaluate_task(wait_for=[row_count])

    # Step 4: train (waits for evaluate, since it reads best_models.json)
    train_task(wait_for=[evaluate_results])

    logger.info("quarterly-retrain finished")


def main():
    parser = argparse.ArgumentParser(description="Run the quarterly-retrain Prefect flow.")
    parser.add_argument("--source", choices=["api", "csv"], default="api")
    parser.add_argument("--dbt-target", default="dev")
    args = parser.parse_args()
    quarterly_retrain_flow(source=args.source, dbt_target=args.dbt_target)


if __name__ == "__main__":
    main()