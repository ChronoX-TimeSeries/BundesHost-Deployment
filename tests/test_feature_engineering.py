"""Tests for bundeshost.modeling.feature_engineering."""

import pandas as pd

from bundeshost.modeling.feature_engineering import build_state_series, create_corona_dummy

# ==================================================
# build_state_series
# ==================================================


def test_build_state_series_returns_series(sample_tourism_df, sample_state):
    """build_state_series should return a pandas Series."""
    series = build_state_series(sample_tourism_df, sample_state)
    assert isinstance(series, pd.Series)


def test_build_state_series_correct_state(sample_tourism_df, sample_state):
    """The series should contain only data for the requested state."""
    series = build_state_series(sample_tourism_df, sample_state)
    # Our fixture has 36 monthly rows per state
    assert len(series) == 36


def test_build_state_series_has_datetime_index(sample_tourism_df, sample_state):
    """The returned series should be indexed by date."""
    series = build_state_series(sample_tourism_df, sample_state)
    # Either a DatetimeIndex or a PeriodIndex is acceptable
    assert isinstance(series.index, (pd.DatetimeIndex, pd.PeriodIndex))


# ==================================================
# create_corona_dummy
# ==================================================


def test_corona_dummy_zero_before_covid():
    """Dummy should be 0 for dates before COVID start (March 2020)."""
    idx = pd.date_range("2019-01-01", "2019-12-01", freq="MS")
    dummy = create_corona_dummy(idx)
    assert (dummy == 0).all()


def test_corona_dummy_one_during_covid():
    """Dummy should be 1 during COVID period (March 2020 – May 2021)."""
    idx = pd.date_range("2020-06-01", "2021-03-01", freq="MS")
    dummy = create_corona_dummy(idx)
    assert (dummy == 1).all()


def test_corona_dummy_zero_after_covid():
    """Dummy should be 0 after COVID end (May 2021)."""
    idx = pd.date_range("2022-01-01", "2022-12-01", freq="MS")
    dummy = create_corona_dummy(idx)
    assert (dummy == 0).all()
