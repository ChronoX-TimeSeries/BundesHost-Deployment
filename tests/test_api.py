"""Tests for the BundesHost FastAPI service."""

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