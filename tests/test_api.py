"""Tests for the BundesHost FastAPI service."""

import pytest
from fastapi.testclient import TestClient

from bundeshost.api.main import app

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_shape():
    response = client.get("/health")
    assert response.json() == {"status": "ok"}


def test_states_returns_200():
    response = client.get("/states")
    assert response.status_code == 200


def test_states_returns_16_states():
    response = client.get("/states")
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 16
    assert "Bayern" in data
    assert "Hamburg" in data


# ==================================================
# /forecast/{state}
# ==================================================


@pytest.mark.integration
def test_forecast_returns_200_for_valid_state():
    response = client.get("/forecast/Berlin?horizon=12")
    assert response.status_code == 200


@pytest.mark.integration
def test_forecast_response_shape():
    response = client.get("/forecast/Berlin?horizon=6")
    data = response.json()
    assert data["state"] == "Berlin"
    assert data["horizon"] == 6
    assert "today" in data
    assert "last_training_date" in data
    assert isinstance(data["forecast"], list)
    assert len(data["forecast"]) > 0


@pytest.mark.integration
def test_forecast_point_has_required_fields():
    response = client.get("/forecast/Hamburg?horizon=3")
    point = response.json()["forecast"][0]
    assert set(point.keys()) == {"date", "forecast", "lower_ci", "upper_ci", "type"}
    assert point["type"] in {"backfill", "future"}


@pytest.mark.integration
def test_forecast_has_both_backfill_and_future():
    response = client.get("/forecast/Berlin?horizon=12")
    types = {p["type"] for p in response.json()["forecast"]}
    # Today is past the model's last training date, so we expect both buckets.
    assert "backfill" in types
    assert "future" in types


def test_forecast_returns_404_for_unknown_state():
    response = client.get("/forecast/Atlantis?horizon=12")
    assert response.status_code == 404


def test_forecast_rejects_horizon_too_small():
    response = client.get("/forecast/Berlin?horizon=0")
    assert response.status_code == 422


def test_forecast_rejects_horizon_too_large():
    response = client.get("/forecast/Berlin?horizon=99")
    assert response.status_code == 422


# ==================================================
# /history/{state}
# ==================================================


@pytest.mark.integration
def test_history_returns_200_for_valid_state():
    response = client.get("/history/Berlin?last_n=36")
    assert response.status_code == 200


@pytest.mark.integration
def test_history_response_shape():
    response = client.get("/history/Berlin?last_n=12")
    data = response.json()
    assert data["state"] == "Berlin"
    assert isinstance(data["history"], list)
    assert len(data["history"]) == 12


@pytest.mark.integration
def test_history_point_has_required_fields():
    response = client.get("/history/Hamburg?last_n=3")
    point = response.json()["history"][0]
    assert set(point.keys()) == {"date", "arrivals"}
    assert isinstance(point["arrivals"], float)


def test_history_returns_404_for_unknown_state():
    response = client.get("/history/Atlantis?last_n=12")
    assert response.status_code == 404


def test_history_rejects_invalid_last_n():
    response = client.get("/history/Berlin?last_n=0")
    assert response.status_code == 422


# ==================================================
# /summary
# ==================================================


@pytest.mark.integration
def test_summary_returns_200():
    response = client.get("/summary")
    assert response.status_code == 200


@pytest.mark.integration
def test_summary_response_shape():
    response = client.get("/summary")
    data = response.json()
    assert "as_of" in data
    assert isinstance(data["states"], list)
    assert len(data["states"]) == 16


@pytest.mark.integration
def test_summary_state_item_has_required_fields():
    response = client.get("/summary")
    item = response.json()["states"][0]
    assert set(item.keys()) == {"state", "latest_arrivals"}
    assert isinstance(item["latest_arrivals"], float)
