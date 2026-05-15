import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .config import BASE_DIR


# ==================================================
# Load tourism dataset (Destatis)
# ==================================================

def get_tourism_data():

    path = BASE_DIR / "data" / "raw" / "tourism_monthly_aggregated.csv"

    # Read CSV skipping metadata
    df_raw = pd.read_csv(
        path,
        sep=";",
        skiprows=7,
        header=None,
        encoding="utf-8"
    )

    # Forward fill year column
    df_raw[0] = df_raw[0].ffill()

    # Keep only value columns
    cols_to_keep = [0, 1] + list(range(2, df_raw.shape[1], 2))
    df_step1 = df_raw.iloc[:, cols_to_keep].copy()

    # Read state names from header
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    state_line = lines[4]
    state_parts = state_line.strip().split(";")
    states = [s for s in state_parts if s != ""]

    # Create column names
    new_columns = ["year", "month"]

    for state in states:
        new_columns.append(f"{state}_arrivals")
        new_columns.append(f"{state}_overnight")

    df_step1.columns = new_columns

    # Create datetime column
    df_step1["date"] = pd.to_datetime(
        df_step1["year"].astype(str) + " " + df_step1["month"],
        format="%Y %B"
    )

    # Move date to first column
    cols = df_step1.columns.tolist()
    cols.remove("date")
    df_step1 = df_step1[["date"] + cols]

    # Drop year/month
    df_step1 = df_step1.drop(columns=["year", "month"])

    # Sort by date
    df_step1 = df_step1.sort_values("date").reset_index(drop=True)

    # Wide → long
    df_long = df_step1.melt(
        id_vars="date",
        var_name="state_metric",
        value_name="value"
    )

    # Split state / metric
    df_long[["state", "metric"]] = df_long["state_metric"].str.rsplit(
        "_",
        n=1,
        expand=True
    )

    # Pivot
    df_long = df_long.pivot_table(
        index=["date", "state"],
        columns="metric",
        values="value"
    ).reset_index()

    # Sort
    df_long = df_long.sort_values(["state", "date"]).reset_index(drop=True)

    return df_long

