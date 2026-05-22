"""Destatis tourism data client.

Two source paths:
- fetch_from_csv: read the historical CSV bundled with the repo (one-time seed)
- fetch_from_api: query the Destatis GENESIS-Online API (Phase E)

Both return the same long-format DataFrame:
    columns: date (datetime), state (str), arrivals (float), overnight (float)
"""
import os 

import pandas as pd
from dotenv import load_dotenv


from ..config import BASE_DIR

load_dotenv()

def fetch_from_csv(path=None):
    """Read the historical Destatis CSV and return a long-format DataFrame.

    Migrated from data_pipeline.get_tourism_data() — same logic.
    """
    if path is None:
        path = BASE_DIR / "data" / "raw" / "tourism_monthly_aggregated.csv"

    df_raw = pd.read_csv(path, sep=";", skiprows=7, header=None, encoding="utf-8")

    df_raw[0] = df_raw[0].ffill()

    cols_to_keep = [0, 1] + list(range(2, df_raw.shape[1], 2))
    df_step1 = df_raw.iloc[:, cols_to_keep].copy()

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    state_line = lines[4]
    state_parts = state_line.strip().split(";")
    states = [s for s in state_parts if s != ""]

    new_columns = ["year", "month"]
    for state in states:
        new_columns.append(f"{state}_arrivals")
        new_columns.append(f"{state}_overnight")
    df_step1.columns = new_columns

    df_step1["date"] = pd.to_datetime(
        df_step1["year"].astype(str) + " " + df_step1["month"], format="%Y %B"
    )

    cols = df_step1.columns.tolist()
    cols.remove("date")
    df_step1 = df_step1[["date"] + cols]
    df_step1 = df_step1.drop(columns=["year", "month"])
    df_step1 = df_step1.sort_values("date").reset_index(drop=True)

    df_long = df_step1.melt(id_vars="date", var_name="state_metric", value_name="value")
    df_long[["state", "metric"]] = df_long["state_metric"].str.rsplit("_", n=1, expand=True)
    df_long = df_long.pivot_table(
        index=["date", "state"], columns="metric", values="value"
    ).reset_index()
    df_long = df_long.sort_values(["state", "date"]).reset_index(drop=True)

    return df_long[["date", "state", "arrivals", "overnight"]]


def fetch_from_api(start_year: int = 1992) -> pd.DataFrame:
    """Fetch fresh tourism data from the Destatis GENESIS-Online API.

    Queries table 45412-0025 (monthly tourism survey) and returns the same
    long-format DataFrame shape as fetch_from_csv():
        columns: date (datetime), state (str), arrivals (float), overnight (float)

    Args:
        start_year: First year to include. Default 1992 matches the historical CSV.

    Returns:
        DataFrame with one row per (date, state). Rows where both metrics are
        NaN (e.g. months Destatis hasn't published yet) are dropped.

    Raises:
        RuntimeError: if DESTATIS_API_BASE_URL or DESTATIS_API_TOKEN is missing.
        requests.HTTPError: on non-2xx HTTP responses.
        ValueError: if the API returns JSON (an error wrapper) instead of a zip.
    """
    import io
    import zipfile

    import requests


    base_url = os.getenv("DESTATIS_API_BASE_URL")
    token = os.getenv("DESTATIS_API_TOKEN")
    if not base_url or not token:
        raise RuntimeError(
            "DESTATIS_API_BASE_URL and DESTATIS_API_TOKEN must be set in .env"
        )
    if not base_url.endswith("/"):
        base_url += "/"

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "username": token,
        "password": "",
    }
    response = requests.post(
        base_url + "data/tablefile",
        headers=headers,
        data={
            "name": "45412-0025",
            "startyear": start_year,
            "compress": "true",
            "format": "ffcsv",
            "language": "en",
        },
        timeout=120,
    )
    response.raise_for_status()

    # Destatis sometimes wraps errors in a 200 JSON body. Sniff first byte.
    if response.content[:1] == b"{":
        raise ValueError(
            f"Destatis returned JSON instead of a zip: {response.text[:300]}"
        )

    # Unzip in memory; archive contains a single ffcsv file.
    zf = zipfile.ZipFile(io.BytesIO(response.content))
    csv_bytes = zf.read(zf.namelist()[0])

    raw = pd.read_csv(
        io.BytesIO(csv_bytes),
        delimiter=";",
        decimal=",",
        na_values=["...", ".", "-", "/", "x"],
        low_memory=False,
    )

    # Each row is one (year, month, state, metric, value). Pivot to wide.
    # MONAT01..MONAT12 -> month integer 1..12
    raw["month"] = raw["1_variable_attribute_code"].str.replace("MONAT", "").astype(int)
    raw["date"] = pd.to_datetime(
        dict(year=raw["time"], month=raw["month"], day=1)
    )
    raw = raw.rename(columns={"2_variable_attribute_label": "state"})

    # GAST01 = arrivals, GAST02 = overnight stays
    metric_map = {"GAST01": "arrivals", "GAST02": "overnight"}
    raw["metric"] = raw["value_variable_code"].map(metric_map)
    if raw["metric"].isna().any():
        unknown = raw.loc[raw["metric"].isna(), "value_variable_code"].unique()
        raise ValueError(f"Unknown value_variable_code(s) from API: {unknown}")

    wide = raw.pivot_table(
        index=["date", "state"], columns="metric", values="value"
    ).reset_index()
    wide.columns.name = None

    # Drop rows where both metrics are NaN (months not yet published).
    wide = wide.dropna(subset=["arrivals", "overnight"], how="all")

    wide = wide.sort_values(["state", "date"]).reset_index(drop=True)
    return wide[["date", "state", "arrivals", "overnight"]]