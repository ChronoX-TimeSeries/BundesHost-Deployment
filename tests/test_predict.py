"""Smoke tests for bundeshost.modeling.predict."""

import pandas as pd
import pytest

from bundeshost.modeling.predict import forecast_state


# ==================================================
# forecast_state smoke tests
# ==================================================


def test_forecast_returns_dataframe(sample_state):
    """forecast_state should return a pandas DataFrame."""
    result = forecast_state(sample_state, horizon=6)
    assert isinstance(result, pd.DataFrame)


def test_forecast_has_correct_rows(sample_state):
    """The returned DataFrame should have exactly `horizon` rows."""
    horizon = 6
    result = forecast_state(sample_state, horizon=horizon)
    assert len(result) == horizon


def test_forecast_has_expected_columns(sample_state):
    """The returned DataFrame should have the expected columns."""
    result = forecast_state(sample_state, horizon=6)
    expected = {"date", "forecast", "lower_ci", "upper_ci"}
    assert set(result.columns) == expected


def test_forecast_values_are_positive(sample_state):
    """Forecasted arrivals should be positive (real-world sanity check)."""
    result = forecast_state(sample_state, horizon=6)
    assert (result["forecast"] > 0).all()


def test_forecast_ci_bounds_are_ordered(sample_state):
    """Lower CI should always be <= forecast <= Upper CI."""
    result = forecast_state(sample_state, horizon=6)
    assert (result["lower_ci"] <= result["forecast"]).all()
    assert (result["forecast"] <= result["upper_ci"]).all()


def test_forecast_raises_on_unknown_state():
    """An invalid state name should raise an error."""
    with pytest.raises((KeyError, FileNotFoundError)):
        forecast_state("Atlantis", horizon=6)