"""Tests for bundeshost.pipelines tasks and flows.

These are mostly smoke tests (verify imports + signatures) plus targeted
mocked tests for the new logic in Phase E (conditional check, cache
invalidation). The heavy tasks (ingest, dbt, evaluate, train) are thin
wrappers around already-tested functions and aren't re-tested here.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Smoke tests: all tasks and flows are importable
# ---------------------------------------------------------------------------


def test_tasks_module_imports():
    from bundeshost.pipelines.tasks import (  # noqa: F401
        check_for_new_data_task,
        dbt_task,
        evaluate_task,
        ingest_task,
        invalidate_api_cache_task,
        train_task,
    )


def test_flows_import():
    from bundeshost.pipelines.ingest_flow import ingest_only_flow  # noqa: F401
    from bundeshost.pipelines.quarterly_retrain import quarterly_retrain_flow  # noqa: F401


# ---------------------------------------------------------------------------
# check_for_new_data_task — new logic in Phase E
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_api_df():
    """Two months × two states, latest = 2026-03-01."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-02-01", "2026-02-01", "2026-03-01", "2026-03-01"]),
            "state": ["Hamburg", "Berlin", "Hamburg", "Berlin"],
            "arrivals": [100, 200, 110, 210],
            "overnight": [300, 400, 310, 410],
        }
    )


def test_check_for_new_data_detects_newer_api(fake_api_df):
    """API has 2026-03, DB only has 2026-01 -> has_new_data = True."""
    from bundeshost.pipelines.tasks import check_for_new_data_task

    fake_conn = MagicMock()
    fake_conn.execute.return_value.scalar.return_value = pd.to_datetime("2026-01-01").date()
    fake_engine = MagicMock()
    fake_engine.connect.return_value.__enter__.return_value = fake_conn

    with (
        patch("bundeshost.data.destatis_client.fetch_from_api", return_value=fake_api_df),
        patch("bundeshost.pipelines.tasks.get_engine", return_value=fake_engine),
        patch("bundeshost.pipelines.tasks.get_run_logger"),
    ):
        result = check_for_new_data_task.fn()

    assert result["has_new_data"] is True
    assert result["latest_in_api"] == pd.to_datetime("2026-03-01").date()
    assert result["latest_in_db"] == pd.to_datetime("2026-01-01").date()


def test_check_for_new_data_no_change(fake_api_df):
    """API and DB both end at 2026-03 -> has_new_data = False."""
    from bundeshost.pipelines.tasks import check_for_new_data_task

    fake_conn = MagicMock()
    fake_conn.execute.return_value.scalar.return_value = pd.to_datetime("2026-03-01").date()
    fake_engine = MagicMock()
    fake_engine.connect.return_value.__enter__.return_value = fake_conn

    with (
        patch("bundeshost.data.destatis_client.fetch_from_api", return_value=fake_api_df),
        patch("bundeshost.pipelines.tasks.get_engine", return_value=fake_engine),
        patch("bundeshost.pipelines.tasks.get_run_logger"),
    ):
        result = check_for_new_data_task.fn()

    assert result["has_new_data"] is False


def test_check_for_new_data_empty_db(fake_api_df):
    """DB is empty -> has_new_data = True (we should ingest everything)."""
    from bundeshost.pipelines.tasks import check_for_new_data_task

    fake_conn = MagicMock()
    fake_conn.execute.return_value.scalar.return_value = None
    fake_engine = MagicMock()
    fake_engine.connect.return_value.__enter__.return_value = fake_conn

    with (
        patch("bundeshost.data.destatis_client.fetch_from_api", return_value=fake_api_df),
        patch("bundeshost.pipelines.tasks.get_engine", return_value=fake_engine),
        patch("bundeshost.pipelines.tasks.get_run_logger"),
    ):
        result = check_for_new_data_task.fn()

    assert result["has_new_data"] is True
    assert result["latest_in_db"] is None


# ---------------------------------------------------------------------------
# invalidate_api_cache_task — graceful degradation
# ---------------------------------------------------------------------------


def test_invalidate_api_cache_success():
    """When the API responds 200, return the JSON payload."""
    from bundeshost.pipelines.tasks import invalidate_api_cache_task

    fake_response = MagicMock()
    fake_response.json.return_value = {"status": "cleared", "models_evicted": 3}
    fake_response.raise_for_status = lambda: None

    with (
        patch("httpx.post", return_value=fake_response),
        patch("bundeshost.pipelines.tasks.get_run_logger"),
    ):
        result = invalidate_api_cache_task.fn()

    assert result == {"status": "cleared", "models_evicted": 3}


def test_invalidate_api_cache_unreachable_does_not_raise():
    """When the API is down, log a warning but don't fail the flow."""
    import httpx

    from bundeshost.pipelines.tasks import invalidate_api_cache_task

    with (
        patch("httpx.post", side_effect=httpx.ConnectError("connection refused")),
        patch("bundeshost.pipelines.tasks.get_run_logger"),
    ):
        result = invalidate_api_cache_task.fn()

    assert result["status"] == "unreachable"
    assert "connection refused" in result["error"]


# ---------------------------------------------------------------------------
# API admin endpoint
# ---------------------------------------------------------------------------


def test_admin_clear_cache_endpoint():
    """POST /admin/clear-cache returns the count of evicted models."""
    from fastapi.testclient import TestClient

    from bundeshost.api.main import app
    from bundeshost.modeling.predict import _MODEL_CACHE

    # Seed the cache so we have something to evict
    _MODEL_CACHE["Hamburg"] = ("dummy_model", "Hamburg_sarimax")
    _MODEL_CACHE["Berlin"] = ("dummy_model", "Berlin_sarima")

    client = TestClient(app)
    response = client.post("/admin/clear-cache")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "cleared"
    assert body["models_evicted"] == 2

    # Re-clear: should now be 0
    response2 = client.post("/admin/clear-cache")
    assert response2.json()["models_evicted"] == 0
