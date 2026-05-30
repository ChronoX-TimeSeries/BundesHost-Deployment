"""Tests for bundeshost.registry helpers.

We mock MlflowClient so these tests don't hit DagsHub.
"""

from unittest.mock import MagicMock, patch

import pytest
from mlflow.exceptions import RestException

from bundeshost.registry import (
    PRODUCTION_ALIAS,
    clear_model_cache,
    get_production_mape,
    get_production_version,
    promote_to_production,
    should_promote,
)

# ==================================================
# Fixtures
# ==================================================


@pytest.fixture
def mock_client():
    """A MagicMock that quacks like an MlflowClient."""
    return MagicMock()


def _make_version(version="1", tags=None):
    """Build a fake ModelVersion-like object."""
    mv = MagicMock()
    mv.version = version
    mv.tags = tags or {}
    return mv


def _alias_missing():
    """A RestException as MLflow raises it when an alias does not exist."""
    return RestException({"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "alias not found"})


# ==================================================
# get_production_version
# ==================================================


def test_get_production_version_returns_version_when_exists(mock_client):
    mock_client.get_model_version_by_alias.return_value = _make_version("2")
    result = get_production_version(mock_client, "bundeshost-Hamburg")
    assert result.version == "2"
    mock_client.get_model_version_by_alias.assert_called_once_with(
        "bundeshost-Hamburg", PRODUCTION_ALIAS
    )


def test_get_production_version_returns_none_when_no_alias(mock_client):
    mock_client.get_model_version_by_alias.side_effect = _alias_missing()
    result = get_production_version(mock_client, "bundeshost-Hamburg")
    assert result is None


def test_get_production_version_returns_none_when_model_missing(mock_client):
    mock_client.get_model_version_by_alias.side_effect = _alias_missing()
    result = get_production_version(mock_client, "bundeshost-Atlantis")
    assert result is None


def test_get_production_version_reraises_unexpected_errors(mock_client):
    mock_client.get_model_version_by_alias.side_effect = RestException(
        {"error_code": "INTERNAL_ERROR", "message": "boom"}
    )
    with pytest.raises(RestException):
        get_production_version(mock_client, "bundeshost-Hamburg")


# ==================================================
# get_production_mape
# ==================================================


def test_get_production_mape_reads_tag(mock_client):
    mock_client.get_model_version_by_alias.return_value = _make_version(
        "2", tags={"production_mape": "3.5"}
    )
    assert get_production_mape(mock_client, "bundeshost-Hamburg") == 3.5


def test_get_production_mape_returns_none_when_tag_missing(mock_client):
    mock_client.get_model_version_by_alias.return_value = _make_version("2", tags={})
    assert get_production_mape(mock_client, "bundeshost-Hamburg") is None


def test_get_production_mape_returns_none_when_no_production(mock_client):
    mock_client.get_model_version_by_alias.side_effect = _alias_missing()
    assert get_production_mape(mock_client, "bundeshost-Hamburg") is None


# ==================================================
# should_promote
# ==================================================


def test_should_promote_true_when_no_production(mock_client):
    mock_client.get_model_version_by_alias.side_effect = _alias_missing()
    assert should_promote(mock_client, "bundeshost-Hamburg", new_mape=5.0) is True


def test_should_promote_true_when_production_has_no_mape_tag(mock_client):
    mock_client.get_model_version_by_alias.return_value = _make_version("2", tags={})
    assert should_promote(mock_client, "bundeshost-Hamburg", new_mape=5.0) is True


def test_should_promote_true_when_new_mape_is_better(mock_client):
    mock_client.get_model_version_by_alias.return_value = _make_version(
        "2", tags={"production_mape": "5.0"}
    )
    assert should_promote(mock_client, "bundeshost-Hamburg", new_mape=3.0) is True


def test_should_promote_false_when_new_mape_is_worse(mock_client):
    mock_client.get_model_version_by_alias.return_value = _make_version(
        "2", tags={"production_mape": "3.0"}
    )
    assert should_promote(mock_client, "bundeshost-Hamburg", new_mape=5.0) is False


def test_should_promote_false_when_new_mape_is_equal(mock_client):
    mock_client.get_model_version_by_alias.return_value = _make_version(
        "2", tags={"production_mape": "3.0"}
    )
    # Strictly less-than — equal does not promote.
    assert should_promote(mock_client, "bundeshost-Hamburg", new_mape=3.0) is False


# ==================================================
# promote_to_production
# ==================================================


def test_promote_to_production_sets_alias_and_tags(mock_client):
    promote_to_production(mock_client, "bundeshost-Hamburg", version="3", mape=2.5)

    mock_client.set_registered_model_alias.assert_called_once_with(
        name="bundeshost-Hamburg",
        alias=PRODUCTION_ALIAS,
        version="3",
    )
    mock_client.set_model_version_tag.assert_called_once_with(
        name="bundeshost-Hamburg",
        version="3",
        key="production_mape",
        value="2.5",
    )


# ==================================================
# clear_model_cache
# ==================================================


def test_clear_model_cache_empties_predict_cache():
    # Seed the cache with a fake entry, then clear it.
    with patch("bundeshost.modeling.predict._MODEL_CACHE", {"Hamburg": ("model", "name")}):
        from bundeshost.modeling import predict as predict_module

        assert "Hamburg" in predict_module._MODEL_CACHE
        clear_model_cache()
        assert predict_module._MODEL_CACHE == {}
