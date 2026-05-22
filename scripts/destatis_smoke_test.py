"""Smoke test for Destatis GENESIS-Online REST API (Phase E, Step 1).

Goal: verify token + connectivity, then fetch table 45412-0025 (tourism)
and print enough of the response to understand its shape.

This script does NOT touch destatis_client.py or ingest.py.
It writes raw artifacts to /tmp so we can inspect them by hand:
  - /tmp/destatis_logincheck.json   (auth test)
  - /tmp/destatis_45412-0025.csv    (unzipped ffcsv response)
"""

import io
import os
import sys
import zipfile

import pandas as pd
import requests
from dotenv import load_dotenv


def main():
    load_dotenv()

    base_url = os.getenv("DESTATIS_API_BASE_URL")
    token = os.getenv("DESTATIS_API_TOKEN")

    if not base_url or not token:
        print("ERROR: DESTATIS_API_BASE_URL or DESTATIS_API_TOKEN not set in .env")
        sys.exit(1)

    if not base_url.endswith("/"):
        base_url += "/"

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "username": token,
        "password": "",
    }

    # 1. Logincheck — cheapest sanity check.
    print("=" * 60)
    print("STEP 1: helloworld/logincheck")
    print("=" * 60)
    r = requests.post(
        base_url + "helloworld/logincheck",
        headers=headers,
        data={"language": "en"},
        timeout=30,
    )
    print(f"HTTP status: {r.status_code}")
    print(f"Response (first 300 chars): {r.text[:300]}")
    with open("/tmp/destatis_logincheck.json", "w") as f:
        f.write(r.text)
    print("Wrote /tmp/destatis_logincheck.json")

    if r.status_code != 200:
        print("Logincheck failed, aborting.")
        sys.exit(1)

    # 2. Fetch table 45412-0025 as ffcsv (zipped).
    print()
    print("=" * 60)
    print("STEP 2: data/tablefile  name=45412-0025  format=ffcsv")
    print("=" * 60)
    r = requests.post(
        base_url + "data/tablefile",
        headers=headers,
        data={
            "name": "45412-0025",
            "compress": "true",
            "format": "ffcsv",
            "language": "en",
        },
        timeout=120,
    )
    print(f"HTTP status: {r.status_code}")
    print(f"Content-Type: {r.headers.get('Content-Type')}")
    print(f"Content length: {len(r.content)} bytes")

    if r.status_code != 200:
        print(f"Fetch failed. Body: {r.text[:500]}")
        sys.exit(1)

    # The API sometimes returns a JSON error wrapped in 200. Sniff first bytes.
    if r.content[:1] == b"{":
        print("Response looks like JSON, not a zip. Body:")
        print(r.text[:500])
        sys.exit(1)

    # 3. Unzip in memory.
    print()
    print("=" * 60)
    print("STEP 3: unzip + peek")
    print("=" * 60)
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    print(f"Files in zip: {names}")
    csv_name = names[0]
    csv_bytes = zf.read(csv_name)
    print(f"CSV size: {len(csv_bytes)} bytes")

    with open("/tmp/destatis_45412-0025.csv", "wb") as f:
        f.write(csv_bytes)
    print("Wrote /tmp/destatis_45412-0025.csv")

    # 4. Parse with pandas to see the shape.
    print()
    print("=" * 60)
    print("STEP 4: pandas peek")
    print("=" * 60)
    df = pd.read_csv(
        io.BytesIO(csv_bytes),
        delimiter=";",
        decimal=",",
        na_values=["...", ".", "-", "/", "x"],
        low_memory=False,
    )
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print()
    print("First 5 rows:")
    print(df.head().to_string())
    print()
    print("Last 5 rows:")
    print(df.tail().to_string())
    print()
    print("dtypes:")
    print(df.dtypes)


if __name__ == "__main__":
    main()