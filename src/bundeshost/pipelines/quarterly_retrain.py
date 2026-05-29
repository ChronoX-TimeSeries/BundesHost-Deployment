"""Quarterly retrain flow.

Wires the four tasks together in dependency order, with an early-exit
when Destatis has no new data:

    check_for_new_data
       ├─ if no new data → return early
       └─ if new data    → ingest -> dbt -> evaluate -> train

What still needs to be added in later steps:
- Step 7: cron schedule
- Step 8: notification on completion

Run locally:
    python -m bundeshost.pipelines.quarterly_retrain
    python -m bundeshost.pipelines.quarterly_retrain --force
"""

import argparse

from prefect import flow, get_run_logger

from bundeshost.pipelines.tasks import (
    check_for_new_data_task,
    dbt_task,
    drift_report_task,
    evaluate_task,
    ingest_task,
    invalidate_api_cache_task,
    train_task,
)


@flow(name="quarterly-retrain")
def quarterly_retrain_flow(force: bool = False, dbt_target: str = "dev") -> dict:
    """End-to-end retrain pipeline with conditional execution.

    Parameters
    ----------
    force : bool
        If True, run the full pipeline even when Destatis has no new data.
        Useful for manual reruns and testing.
    dbt_target : str
        dbt target profile (default "dev").

    Returns
    -------
    dict with keys:
        ran (bool): whether the heavy steps (ingest/dbt/evaluate/train) ran
        latest_in_db (date or None): max(date) in Postgres before this run
        latest_in_api (date): max(date) Destatis returned
    """
    logger = get_run_logger()
    logger.info(f"Starting quarterly-retrain (force={force}, dbt_target={dbt_target})")

    # Step 1: check if Destatis has new data
    check = check_for_new_data_task()

    if not check["has_new_data"] and not force:
        logger.info(
            f"No new data from Destatis "
            f"(DB latest: {check['latest_in_db']}, API latest: {check['latest_in_api']}). "
            f"Skipping ingest/dbt/evaluate/train. Re-run with --force to override."
        )
        return {
            "ran": False,
            "latest_in_db": check["latest_in_db"],
            "latest_in_api": check["latest_in_api"],
        }

    if check["has_new_data"]:
        logger.info(
            f"New data detected: API has up to {check['latest_in_api']}, "
            f"DB only had up to {check['latest_in_db']}"
        )
    else:
        logger.info("force=True; running full pipeline regardless of new-data check")

    # Step 2: ingest (reuse the DataFrame we already fetched)
    row_count = ingest_task(df=check["df"])

    # Step 3: dbt
    dbt_task(target=dbt_target, wait_for=[row_count])

    # Step 3.5: drift report (soft-fail, monitoring only)
    drift_report_task(wait_for=[row_count])

    # Step 4: evaluate
    evaluate_results = evaluate_task(wait_for=[row_count])

    # Step 5: train
    train_task(wait_for=[evaluate_results])

    # Step 6: tell the API to clear its model cache so the new models are served
    invalidate_api_cache_task()

    logger.info("quarterly-retrain finished")
    return {
        "ran": True,
        "latest_in_db": check["latest_in_db"],
        "latest_in_api": check["latest_in_api"],
    }


def main():
    parser = argparse.ArgumentParser(description="Run the quarterly-retrain Prefect flow.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run the full pipeline even if Destatis has no new data.",
    )
    parser.add_argument("--dbt-target", default="dev")
    args = parser.parse_args()
    quarterly_retrain_flow(force=args.force, dbt_target=args.dbt_target)


if __name__ == "__main__":
    main()
