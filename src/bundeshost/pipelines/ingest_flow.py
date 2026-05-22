"""Prefect flow that ingests tourism data into Postgres.

Wraps bundeshost.data.ingest as a Prefect task so we get:
- Run history visible in Prefect Cloud
- Automatic retries on failure
- Logging tied to a flow run

Run locally:
    python -m bundeshost.pipelines.ingest_flow
    python -m bundeshost.pipelines.ingest_flow --source csv
"""

import argparse

from prefect import flow, get_run_logger, task

from bundeshost.data.ingest import get_engine, load_data, upsert_tourism_raw


@task(retries=2, retry_delay_seconds=10)
def ingest_task(source: str = "api") -> int:
    """Fetch tourism data and UPSERT it into public.tourism_raw.

    Returns the total row count in the table after the upsert.
    """
    logger = get_run_logger()
    logger.info(f"Ingest task started (source={source})")

    df = load_data(source)
    logger.info(f"Loaded {len(df)} rows from {source}")

    engine = get_engine()
    total = upsert_tourism_raw(df, engine)

    logger.info(f"UPSERT complete; tourism_raw now has {total} rows")
    return total


@flow(name="ingest-only")
def ingest_only_flow(source: str = "api") -> int:
    """One-task flow that just runs ingest. Used to verify Prefect plumbing."""
    logger = get_run_logger()
    logger.info(f"Starting ingest-only flow with source={source}")
    total = ingest_task(source=source)
    logger.info(f"Flow finished. Final row count: {total}")
    return total


def main():
    parser = argparse.ArgumentParser(description="Run the ingest-only Prefect flow.")
    parser.add_argument("--source", choices=["api", "csv"], default="api")
    args = parser.parse_args()
    ingest_only_flow(source=args.source)


if __name__ == "__main__":
    main()