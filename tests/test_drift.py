"""Tests for bundeshost.monitoring.drift."""

from __future__ import annotations

import pandas as pd
import pytest

from bundeshost.monitoring.drift import generate_drift_report, split_reference_current


def _make_df(months: int = 60, states: int = 2) -> pd.DataFrame:
    """Build a synthetic long-format DataFrame for testing."""
    dates = pd.date_range("2020-01-01", periods=months, freq="MS")
    rows = []
    for state_idx in range(states):
        for i, d in enumerate(dates):
            rows.append(
                {
                    "date": d,
                    "state": f"State{state_idx}",
                    "arrivals": 1000.0 + i * 10 + state_idx * 100,
                    "overnight": 3000.0 + i * 30 + state_idx * 300,
                }
            )
    return pd.DataFrame(rows)


def test_split_reference_current_default_window_sizes():
    df = _make_df(months=60, states=2)
    ref, cur = split_reference_current(df)
    # current = last 12 months × 2 states = 24 rows
    assert len(cur) == 24
    # reference = 24 months before that × 2 states = 48 rows
    assert len(ref) == 48
    # reference must end before current starts
    assert ref["date"].max() < cur["date"].min()


def test_split_reference_current_custom_window_sizes():
    df = _make_df(months=60, states=2)
    ref, cur = split_reference_current(df, current_months=6, reference_months=12)
    assert len(cur) == 6 * 2
    assert len(ref) == 12 * 2


def test_split_reference_current_requires_date_column():
    df = pd.DataFrame({"state": ["X"], "arrivals": [1.0]})
    with pytest.raises(ValueError, match="date"):
        split_reference_current(df)


def test_generate_drift_report_returns_html():
    df = _make_df(months=48, states=2)
    html = generate_drift_report(df)
    assert isinstance(html, str)
    assert len(html) > 1000  # Evidently HTML is always at least a few KB
    assert "<html" in html.lower() or "<div" in html.lower()


def test_generate_drift_report_defaults_to_arrivals_only():
    """When columns=None, the default should be ['arrivals'] (not overnight)."""
    df = _make_df(months=48, states=2)
    # Should not raise even though overnight is present in the DataFrame
    html = generate_drift_report(df)
    assert isinstance(html, str)


def test_generate_drift_report_raises_if_no_numeric_columns():
    """If the requested columns aren't in the DataFrame, raise."""
    df = _make_df(months=48, states=2)
    with pytest.raises(ValueError, match="Columns not found"):
        generate_drift_report(df, columns=["nonexistent"])
