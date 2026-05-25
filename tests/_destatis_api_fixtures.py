"""Helpers to build fake Destatis API responses for testing.

The real Destatis API returns a gzipped ffcsv with a specific long-format shape.
These helpers build minimal in-memory equivalents so we can mock requests.post
without hitting the real API or needing credentials.
"""

import io
import zipfile

import pandas as pd


def build_ffcsv_bytes(rows: list[dict]) -> bytes:
    """Serialize a list of row dicts into ffcsv bytes (German conventions).

    Each row dict must have the columns the parser reads:
        time, 1_variable_attribute_code, 2_variable_attribute_label,
        value_variable_code, value
    Extra columns are fine.
    """
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, sep=";", decimal=",", index=False)
    return buf.getvalue().encode("utf-8")


def build_zip_response(rows: list[dict], filename: str = "45412-0025_test.csv") -> bytes:
    """Wrap ffcsv bytes in a zip archive, matching the real API response shape."""
    csv_bytes = build_ffcsv_bytes(rows)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(filename, csv_bytes)
    return buf.getvalue()


def make_row(year: int, month: int, state: str, metric_code: str, value) -> dict:
    """Build one row in the shape Destatis returns.

    metric_code is 'GAST01' (arrivals) or 'GAST02' (overnight stays).
    value can be a number or 'NaN'-like ('...', '.', '-') for missing.
    """
    return {
        "statistics_code": 45412,
        "statistics_label": "Monthly tourism survey",
        "time_code": "JAHR",
        "time_label": "Year",
        "time": year,
        "1_variable_code": "MONAT",
        "1_variable_label": "Months",
        "1_variable_attribute_code": f"MONAT{month:02d}",
        "1_variable_attribute_label": pd.Timestamp(year=year, month=month, day=1).strftime("%B"),
        "2_variable_code": "DLAND",
        "2_variable_label": "Länder",
        "2_variable_attribute_code": 1,
        "2_variable_attribute_label": state,
        "value": value,
        "value_unit": "number",
        "value_variable_code": metric_code,
        "value_variable_label": "Arrivals" if metric_code == "GAST01" else "Overnight stays",
    }
