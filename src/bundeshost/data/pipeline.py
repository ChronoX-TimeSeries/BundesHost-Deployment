"""Read modeling-ready tourism data from the dbt mart.

Replaces the legacy CSV-parsing pipeline. The CSV path is preserved
in destatis_client.fetch_from_csv() for one-time historical seeding.
"""

import os

import pandas as pd
from sqlalchemy import create_engine, text


def _get_engine():
    """Build a SQLAlchemy engine from POSTGRES_* env vars."""
    user = os.getenv("POSTGRES_USER", "bundeshost")
    password = os.getenv("POSTGRES_PASSWORD", "bundeshost")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "bundeshost")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)


def get_tourism_data():
    """Return the modeling-ready monthly tourism data.

    Reads from {DBT_MART_SCHEMA}.fct_tourism_monthly. The schema name is
    configurable via env var so dev/prod can point at different schemas.
    """
    schema = os.getenv("DBT_MART_SCHEMA", "dbt_dev_mart")
    engine = _get_engine()
    query = text(f"SELECT date, state, arrivals, overnight FROM {schema}.fct_tourism_monthly")
    df = pd.read_sql(query, engine, parse_dates=["date"])
    return df.sort_values(["state", "date"]).reset_index(drop=True)