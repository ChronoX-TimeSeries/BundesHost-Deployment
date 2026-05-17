"""Shared pytest fixtures for BundesHost tests."""

import pandas as pd
import pytest


@pytest.fixture
def sample_state():
    """A valid German state name used across tests."""
    return "Hamburg"


@pytest.fixture
def sample_tourism_df():
    """
    A small, predictable DataFrame that mimics the tourism dataset shape.

    Contains 36 monthly observations (2020-2022) for two states,
    so feature engineering can build a real time series from it.
    """
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")

    rows = []
    for state in ["Hamburg", "Berlin"]:
        for i, date in enumerate(dates):
            rows.append({
                "date": date,
                "state": state,
                "arrivals": 100_000 + i * 1_000,  # simple upward trend
            })

    return pd.DataFrame(rows)

@pytest.fixture(scope="session")
def destatis_long_df():
    """The real Destatis CSV, parsed once per test session."""
    from bundeshost.data.destatis_client import fetch_from_csv
    return fetch_from_csv()