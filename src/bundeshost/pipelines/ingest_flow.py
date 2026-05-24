"""Prefect flow that ingests tourism data into Postgres.

Wraps the ingest_task from bundeshost.pipelines.tasks so we get:
- Run history visible in Prefect Cloud
- Automatic retries on failure
- Logging tied to a flow run

Run locally:
    python -m bundeshost.pipelines.ingest_flow
    python -m bundeshost.pipelines.ingest_flow --source csv
"""

import argparse

from prefect import flow, get_run_logger

from bundeshost.pipelines.tasks import ingest_task


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