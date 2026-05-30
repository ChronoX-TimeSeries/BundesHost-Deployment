"""Model quality tests for MLflow Production models.

Runs on CI with MLflow secrets. Fails if any of the 16 states is missing
a '@production' model, or if any production MAPE exceeds the threshold.
"""

import pytest
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

from bundeshost.config import STATES
from bundeshost.registry import PRODUCTION_ALIAS

MAPE_THRESHOLD = 10.0


@pytest.fixture(scope="module")
def client():
    return MlflowClient()


def _get_production(client, name):
    """Return the @production version, or None if the alias is unset."""
    try:
        return client.get_model_version_by_alias(name, PRODUCTION_ALIAS)
    except RestException:
        return None


@pytest.mark.model_quality
@pytest.mark.parametrize("state", STATES)
def test_state_has_production_version(client, state):
    """Every state must have a @production model in the registry."""
    name = f"bundeshost-{state}"
    version = _get_production(client, name)
    assert version is not None, f"No @production version for {name}"


@pytest.mark.model_quality
@pytest.mark.parametrize("state", STATES)
def test_state_production_mape_under_threshold(client, state):
    """Every state's @production model must have MAPE under threshold."""
    name = f"bundeshost-{state}"
    version = _get_production(client, name)
    assert version is not None, f"No @production version for {name}"

    mape_tag = version.tags.get("production_mape")
    assert mape_tag is not None, f"{name} v{version.version} has no production_mape tag"

    mape = float(mape_tag)
    assert (
        mape < MAPE_THRESHOLD
    ), f"{name} v{version.version} MAPE={mape:.2f}% exceeds {MAPE_THRESHOLD}%"
