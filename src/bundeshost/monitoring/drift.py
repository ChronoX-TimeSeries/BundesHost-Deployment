"""Data drift detection using Evidently.

Compares the most recent window of tourism data against an earlier reference
window. Produces an HTML report that can be logged as an MLflow artifact.

This is monitoring, not decision-making — the report flags when the input
distribution has shifted, but does not trigger retraining or promotion.
"""

from __future__ import annotations

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


def split_reference_current(
    df: pd.DataFrame,
    current_months: int = 12,
    reference_months: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a long-format tourism DataFrame into reference and current windows.

    current  = the most recent `current_months` months
    reference = the `reference_months` months immediately preceding `current`

    Parameters
    ----------
    df : DataFrame with at minimum a `date` column (datetime-like).
    current_months : size of the current window in months (default 12).
    reference_months : size of the reference window in months (default 24).

    Returns
    -------
    (reference_df, current_df)
    """
    if "date" not in df.columns:
        raise ValueError("Input DataFrame must have a 'date' column")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    max_date = df["date"].max()
    current_start = max_date - pd.DateOffset(months=current_months - 1)
    reference_end = current_start - pd.DateOffset(months=1)
    reference_start = reference_end - pd.DateOffset(months=reference_months - 1)

    reference_df = df[(df["date"] >= reference_start) & (df["date"] <= reference_end)]
    current_df = df[df["date"] >= current_start]

    return reference_df, current_df


def generate_drift_report(
    df: pd.DataFrame,
    current_months: int = 12,
    reference_months: int = 24,
    columns: list[str] | None = None,
) -> str:
    """Generate an Evidently data-drift report as an HTML string.

    Parameters
    ----------
    df : long-format DataFrame with columns date, state, arrivals, overnight.
    current_months : size of the current window (default 12).
    reference_months : size of the reference window (default 24).
    columns : which numeric columns to check for drift. Defaults to
        ['arrivals', 'overnight'] if both are present.

    Returns
    -------
    HTML string. Caller is responsible for writing it to disk or logging it
    to MLflow.
    """
    reference_df, current_df = split_reference_current(
        df, current_months=current_months, reference_months=reference_months
    )

    if columns is None:
        candidates = ["arrivals"]
        columns = [c for c in candidates if c in df.columns]

    if not columns:
        raise ValueError("No numeric columns to check for drift")

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    ref_slim = reference_df[columns].reset_index(drop=True)
    cur_slim = current_df[columns].reset_index(drop=True)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_slim, current_data=cur_slim)

    return report.get_html()
