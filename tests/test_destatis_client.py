"""Tests for bundeshost.data.destatis_client."""

import pandas as pd
import pytest

from bundeshost.data.destatis_client import fetch_from_api, fetch_from_csv


def test_fetch_from_csv_returns_dataframe(destatis_long_df):
    assert isinstance(destatis_long_df, pd.DataFrame)


def test_fetch_from_csv_expected_columns(destatis_long_df):
    assert list(destatis_long_df.columns) == ["date", "state", "arrivals", "overnight"]


def test_fetch_from_csv_expected_dtypes(destatis_long_df):
    assert pd.api.types.is_datetime64_any_dtype(destatis_long_df["date"])
    assert pd.api.types.is_object_dtype(destatis_long_df["state"])
    assert pd.api.types.is_numeric_dtype(destatis_long_df["arrivals"])
    assert pd.api.types.is_numeric_dtype(destatis_long_df["overnight"])


def test_fetch_from_csv_has_16_states(destatis_long_df):
    assert destatis_long_df["state"].nunique() == 16


def test_fetch_from_csv_no_duplicates_on_date_state(destatis_long_df):
    assert not destatis_long_df.duplicated(subset=["date", "state"]).any()


def test_fetch_from_api_not_implemented():
    with pytest.raises(NotImplementedError):
        fetch_from_api()