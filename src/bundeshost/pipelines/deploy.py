"""Deploy the quarterly_retrain flow to Prefect Cloud with a quarterly schedule.

This registers the flow as a deployment with a cron schedule (quarterly).
For Hobby tier + managed work pools, code is pulled from GitHub at run time.

Usage:
    python -m bundeshost.pipelines.deploy
"""

from prefect import flow

# Cron: at 02:00 on the 1st day of every 3rd month (Jan, Apr, Jul, Oct).
QUARTERLY_CRON = "0 2 1 */3 *"
WORK_POOL_NAME = "bundeshost-pool"
GITHUB_REPO = "https://github.com/ChronoX-TimeSeries/BundesHost-Deployment.git"
FLOW_ENTRYPOINT = "src/bundeshost/pipelines/quarterly_retrain.py:quarterly_retrain_flow"


def main():
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
            "auto-promote in MLflow if MAPE improves."
        ),
        tags=["phase-2e", "production"],
    )


if __name__ == "__main__":
    main()