"""Tests for bundeshost.data.destatis_client."""

from unittest.mock import patch

import pandas as pd
import pytest

from bundeshost.data.destatis_client import fetch_from_api

from ._destatis_api_fixtures import build_zip_response, make_row

# ---------------------------------------------------------------------------
# fetch_from_csv — historical seed (Phase B)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_fetch_from_csv_returns_dataframe(destatis_long_df):
    assert isinstance(destatis_long_df, pd.DataFrame)


@pytest.mark.integration
def test_fetch_from_csv_expected_columns(destatis_long_df):
    assert list(destatis_long_df.columns) == ["date", "state", "arrivals", "overnight"]


@pytest.mark.integration
def test_fetch_from_csv_expected_dtypes(destatis_long_df):
    assert pd.api.types.is_datetime64_any_dtype(destatis_long_df["date"])
    assert pd.api.types.is_object_dtype(destatis_long_df["state"])
    assert pd.api.types.is_numeric_dtype(destatis_long_df["arrivals"])
    assert pd.api.types.is_numeric_dtype(destatis_long_df["overnight"])


@pytest.mark.integration
def test_fetch_from_csv_has_16_states(destatis_long_df):
    assert destatis_long_df["state"].nunique() == 16


@pytest.mark.integration
def test_fetch_from_csv_no_duplicates_on_date_state(destatis_long_df):
    assert not destatis_long_df.duplicated(subset=["date", "state"]).any()


# ---------------------------------------------------------------------------
# fetch_from_api — real GENESIS-Online client (Phase E, step 2)
#
# All tests mock requests.post so we never hit the real API in CI/local runs.
# ---------------------------------------------------------------------------


@pytest.fixture
def api_env(monkeypatch):
    """Set the env vars fetch_from_api needs."""
    monkeypatch.setenv("DESTATIS_API_BASE_URL", "https://example.test/api/")
    monkeypatch.setenv("DESTATIS_API_TOKEN", "fake-token-1234")


@pytest.fixture
def sample_api_rows():
    """Two months × two states × two metrics = 8 rows, plus one all-NaN month."""
    rows = []
    for year, month in [(2025, 1), (2025, 2)]:
        for state in ["Hamburg", "Berlin"]:
            rows.append(make_row(year, month, state, "GAST01", 100_000 + month))
            rows.append(make_row(year, month, state, "GAST02", 300_000 + month))
    # A future month Destatis hasn't published — both metrics are NaN-like.
    for state in ["Hamburg", "Berlin"]:
        rows.append(make_row(2026, 6, state, "GAST01", "..."))
        rows.append(make_row(2026, 6, state, "GAST02", "..."))
    return rows


def _mock_response(content: bytes, status_code: int = 200):
    """Build a minimal object that quacks like requests.Response."""

    class _R:
        pass

    r = _R()
    r.content = content
    r.status_code = status_code
    r.text = content.decode("utf-8", errors="replace")
    r.raise_for_status = lambda: None
    return r


def test_fetch_from_api_raises_when_env_missing(monkeypatch):
    monkeypatch.delenv("DESTATIS_API_BASE_URL", raising=False)
    monkeypatch.delenv("DESTATIS_API_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="DESTATIS_API"):
        fetch_from_api()


def test_fetch_from_api_raises_on_json_error_body(api_env):
    json_error = b'{"Status":{"Code":98,"Content":"Table too large"}}'
    with patch("requests.post", return_value=_mock_response(json_error)):
        with pytest.raises(ValueError, match="JSON"):
            fetch_from_api()


def test_fetch_from_api_returns_expected_shape(api_env, sample_api_rows):
    zip_bytes = build_zip_response(sample_api_rows)
    with patch("requests.post", return_value=_mock_response(zip_bytes)):
        df = fetch_from_api()
    # 2 months × 2 states = 4 rows; the all-NaN June 2026 entries are dropped.
    assert df.shape == (4, 4)


def test_fetch_from_api_returns_expected_columns(api_env, sample_api_rows):
    zip_bytes = build_zip_response(sample_api_rows)
    with patch("requests.post", return_value=_mock_response(zip_bytes)):
        df = fetch_from_api()
    assert list(df.columns) == ["date", "state", "arrivals", "overnight"]


def test_fetch_from_api_returns_expected_dtypes(api_env, sample_api_rows):
    zip_bytes = build_zip_response(sample_api_rows)
    with patch("requests.post", return_value=_mock_response(zip_bytes)):
        df = fetch_from_api()
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_object_dtype(df["state"])
    assert pd.api.types.is_numeric_dtype(df["arrivals"])
    assert pd.api.types.is_numeric_dtype(df["overnight"])


def test_fetch_from_api_pivots_gast_codes_correctly(api_env, sample_api_rows):
    zip_bytes = build_zip_response(sample_api_rows)
    with patch("requests.post", return_value=_mock_response(zip_bytes)):
        df = fetch_from_api()
    # GAST01 = arrivals, GAST02 = overnight. make_row uses 100_000+month vs 300_000+month.
    hamburg_jan = df[(df["state"] == "Hamburg") & (df["date"] == "2025-01-01")].iloc[0]
    assert hamburg_jan["arrivals"] == 100_001
    assert hamburg_jan["overnight"] == 300_001


def test_fetch_from_api_builds_date_from_year_and_monat_code(api_env, sample_api_rows):
    zip_bytes = build_zip_response(sample_api_rows)
    with patch("requests.post", return_value=_mock_response(zip_bytes)):
        df = fetch_from_api()
    expected = {pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")}
    assert set(df["date"].unique()) == expected


def test_fetch_from_api_drops_rows_where_both_metrics_are_nan(api_env, sample_api_rows):
    zip_bytes = build_zip_response(sample_api_rows)
    with patch("requests.post", return_value=_mock_response(zip_bytes)):
        df = fetch_from_api()
    # June 2026 had "..." for both metrics — should not appear.
    assert (df["date"] == pd.Timestamp("2026-06-01")).sum() == 0


def test_fetch_from_api_raises_on_unknown_gast_code(api_env):
    rows = [make_row(2025, 1, "Hamburg", "GAST99", 123)]
    zip_bytes = build_zip_response(rows)
    with patch("requests.post", return_value=_mock_response(zip_bytes)):
        with pytest.raises(ValueError, match="Unknown value_variable_code"):
            fetch_from_api()
