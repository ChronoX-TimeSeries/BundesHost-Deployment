"""Model quality tests for MLflow Production models.

Runs on CI with MLflow secrets. Fails if any of the 16 states is missing
a Production version, or if any Production MAPE exceeds the threshold.
"""

import pytest
from mlflow.tracking import MlflowClient

from bundeshost.config import STATES

MAPE_THRESHOLD = 10.0


@pytest.fixture(scope="module")
def client():
    return MlflowClient()


@pytest.mark.model_quality
@pytest.mark.parametrize("state", STATES)
def test_state_has_production_version(client, state):
    """Every state must have a Production model in the registry."""
    name = f"bundeshost-{state}"
    versions = client.get_latest_versions(name, stages=["Production"])
    assert len(versions) > 0, f"No Production version for {name}"


@pytest.mark.model_quality
@pytest.mark.parametrize("state", STATES)
def test_state_production_mape_under_threshold(client, state):
    """Every state's Production model must have MAPE under threshold."""
    name = f"bundeshost-{state}"
    versions = client.get_latest_versions(name, stages=["Production"])
    assert len(versions) > 0, f"No Production version for {name}"

    mape_tag = versions[0].tags.get("production_mape")
    assert mape_tag is not None, f"{name} v{versions[0].version} has no production_mape tag"

    mape = float(mape_tag)
    assert mape < MAPE_THRESHOLD, (
        f"{name} v{versions[0].version} MAPE={mape:.2f}% exceeds {MAPE_THRESHOLD}%"
    )
