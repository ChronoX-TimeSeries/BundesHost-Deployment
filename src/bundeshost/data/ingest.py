"""Ingest tourism data into Postgres.

Reads from either:
- destatis_client.fetch_from_csv (historical seed, Phase B)
- destatis_client.fetch_from_api (live API, Phase E)

then UPSERTs into public.tourism_raw.

Idempotent: running twice on the same data is a no-op
(except ingested_at refreshes).

Usage:
    python -m bundeshost.data.ingest                  # default: --source api
    python -m bundeshost.data.ingest --source api     # fetch live from Destatis
    python -m bundeshost.data.ingest --source csv     # read historical CSV
"""

import argparse
import os

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert

from .destatis_client import fetch_from_api, fetch_from_csv


def get_engine():
    """Build a SQLAlchemy engine from POSTGRES_* env vars."""
    user = os.getenv("POSTGRES_USER", "bundeshost")
    password = os.getenv("POSTGRES_PASSWORD", "bundeshost")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "bundeshost")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)


def upsert_tourism_raw(df: pd.DataFrame, engine) -> int:
    """UPSERT a long-format DataFrame into public.tourism_raw.

    Expects columns: date, state, arrivals, overnight.
    Returns the number of rows in the table after the upsert.
    """
    records = df.to_dict(orient="records")

    with engine.begin() as conn:
        from sqlalchemy import MetaData, Table

        metadata = MetaData()
        tourism_raw = Table("tourism_raw", metadata, autoload_with=conn, schema="public")

        stmt = insert(tourism_raw).values(records)
        stmt = stmt.on_conflict_do_update(
            index_elements=["date", "state"],
            set_={
                "arrivals": stmt.excluded.arrivals,
                "overnight": stmt.excluded.overnight,
                "ingested_at": text("NOW()"),
            },
        )
        conn.execute(stmt)

        count = conn.execute(text("SELECT COUNT(*) FROM public.tourism_raw")).scalar()

    return count


def load_data(source: str) -> pd.DataFrame:
    """Dispatch to the right fetch function based on the source flag."""
    if source == "api":
        print("Loading from Destatis API...")
        return fetch_from_api()
    if source == "csv":
        print("Loading from historical CSV...")
        return fetch_from_csv()
    raise ValueError(f"Unknown source: {source!r}. Must be 'api' or 'csv'.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest tourism data into public.tourism_raw (UPSERT, idempotent)."
    )
    parser.add_argument(
        "--source",
        choices=["api", "csv"],
        default="api",
        help="Data source: 'api' (live Destatis, default) or 'csv' (historical seed).",
    )
    return parser.parse_args()


def main():
    """CLI entry: load data from the chosen source and UPSERT into Postgres."""
    args = parse_args()

    df = load_data(args.source)
    print(f"Loaded {len(df)} rows")

    print("Connecting to Postgres...")
    engine = get_engine()

    print("UPSERTing...")
    total = upsert_tourism_raw(df, engine)
    print(f"Done. tourism_raw now has {total} rows.")


if __name__ == "__main__":
    main()
