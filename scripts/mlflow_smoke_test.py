"""Smoke test for MLflow connection to DagsHub.

Runs a tiny dummy run that logs one param and one metric. Verifies that:
1. Environment variables are loaded correctly
2. The MLflow client can authenticate against DagsHub
3. A run actually appears in the DagsHub MLflow UI

Usage:
    source .env
    python scripts/mlflow_smoke_test.py
"""

import os
import sys

import mlflow
from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if not tracking_uri:
        print("ERROR: MLFLOW_TRACKING_URI is not set. Did you `source .env`?")
        return 1
    if not username or not password:
        print("ERROR: MLFLOW_TRACKING_USERNAME or MLFLOW_TRACKING_PASSWORD is not set.")
        return 1

    print(f"Tracking URI: {tracking_uri}")
    print(f"Username:     {username}")
    print("Password:     ***")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("smoke-test")

    with mlflow.start_run(run_name="phase-2d-step1-smoke") as run:
        mlflow.log_param("hello", "world")
        mlflow.log_metric("answer", 42.0)
        print(f"Logged run with id: {run.info.run_id}")

    print("OK. Check the DagsHub MLflow UI to confirm the run appears.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
