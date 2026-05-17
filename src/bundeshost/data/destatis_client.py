"""Destatis tourism data client.

Two source paths:
- fetch_from_csv: read the historical CSV bundled with the repo (one-time seed)
- fetch_from_api: query the Destatis GENESIS-Online API (Phase E)

Both return the same long-format DataFrame:
    columns: date (datetime), state (str), arrivals (float), overnight (float)
"""

import pandas as pd

from ..config import BASE_DIR


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


def fetch_from_api(start_date=None):
    """Fetch fresh tourism data from the Destatis GENESIS-Online API.

    Not implemented yet — Phase E wires this up with httpx + Destatis credentials.
    """
    raise NotImplementedError(
        "fetch_from_api will be implemented in Phase E (orchestration). "
        "For now, use fetch_from_csv() to load the historical dataset."
    )