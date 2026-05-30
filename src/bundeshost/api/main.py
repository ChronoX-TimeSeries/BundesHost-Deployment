"""FastAPI application entrypoint for the BundesHost API."""

import os

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query
from prometheus_fastapi_instrumentator import Instrumentator

from bundeshost.api.schemas import (
    ForecastPoint,
    ForecastResponse,
    HistoryPoint,
    HistoryResponse,
    StateSummary,
    SummaryResponse,
)
from bundeshost.config import STATES
from bundeshost.data.pipeline import get_tourism_data
from bundeshost.modeling.feature_engineering import build_state_series
from bundeshost.modeling.predict import forecast_state, get_last_training_date
from bundeshost.registry import clear_model_cache

app = FastAPI(
    title="BundesHost API",
    description="Tourism forecasting and hosting capacity analysis for German states.",
    version="0.1.0",
)

# Prometheus metrics: instruments every route (request count, latency,
# in-progress, per-handler) and exposes them at GET /metrics for scraping.
Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}


@app.get("/states")
def states() -> list[str]:
    """Return the list of German federal states covered by the API."""
    return STATES


@app.get("/forecast/{state}", response_model=ForecastResponse)
def forecast(
    state: str,
    horizon: int = Query(default=12, ge=1, le=24),
) -> ForecastResponse:
    """
    Return a monthly forecast for the given state.

    The response always covers the period from the month after the last
    training date up to `horizon` months past today. Points before today
    are tagged 'backfill' (model's prediction for months we don't have
    real data for yet). Points from today onwards are tagged 'future'.
    """

    # State validation
    if state not in STATES:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found.")

    # Today (month-aligned to match the model's monthly frequency)
    today = pd.Timestamp.today().normalize().replace(day=1)

    # Last date the model was trained on
    try:
        last_training_date = get_last_training_date(state)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    # Gap in months between last_training_date and today
    gap_months = (today.year - last_training_date.year) * 12 + (
        today.month - last_training_date.month
    )
    gap_months = max(gap_months, 0)

    # Total steps = backfill (gap) + future (horizon)
    total_steps = gap_months + horizon

    # Run the forecast
    try:
        forecast_df = forecast_state(state, total_steps)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    # Latest month for which we already have REAL data in the DB.
    series = build_state_series(get_tourism_data(), state).sort_index()
    max_hist = pd.to_datetime(
        series.index.to_timestamp()[-1]
        if hasattr(series.index, "to_timestamp")
        else series.index[-1]
    )

    def _tag(d: pd.Timestamp) -> str:
        # backfill_validated: model predicted it AND we now have real data
        #   for that month (a live backtest of the model).
        # backfill: predicted, but no real data yet (the true gap).
        # future: today onwards.
        if d <= max_hist:
            return "backfill_validated"
        if d < today:
            return "backfill"
        return "future"

    points = [
        ForecastPoint(
            date=row["date"].date(),
            forecast=float(row["forecast"]),
            lower_ci=float(row["lower_ci"]),
            upper_ci=float(row["upper_ci"]),
            type=_tag(row["date"]),
        )
        for _, row in forecast_df.iterrows()
    ]

    return ForecastResponse(
        state=state,
        horizon=horizon,
        today=today.date(),
        last_training_date=last_training_date.date(),
        forecast=points,
    )


@app.get("/history/{state}", response_model=HistoryResponse)
def history(
    state: str,
    last_n: int = Query(default=36, ge=1, le=480),
) -> HistoryResponse:
    """
    Return the last `last_n` months of historical arrivals for the given state.
    """

    # State validation
    if state not in STATES:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found.")

    # Load data + build the state series (PeriodIndex M)
    df = get_tourism_data()
    series = build_state_series(df, state).sort_index()

    # Take the last N months
    tail = series.tail(last_n)

    # Convert PeriodIndex -> Timestamp -> date
    dates = pd.to_datetime(
        tail.index.to_timestamp() if hasattr(tail.index, "to_timestamp") else tail.index
    )

    points = [HistoryPoint(date=d.date(), arrivals=float(v)) for d, v in zip(dates, tail.values)]

    return HistoryResponse(state=state, history=points)


@app.get("/summary", response_model=SummaryResponse)
def summary() -> SummaryResponse:
    """
    Return the latest available monthly arrivals for every state.

    Used by the frontend for cross-state comparison charts.
    """

    df = get_tourism_data().sort_values("date")

    # Latest available date across the dataset (single value for the whole snapshot)
    as_of = pd.to_datetime(df["date"].max()).date()

    items: list[StateSummary] = []
    for s in STATES:
        state_data = df[df["state"] == s]
        if len(state_data) == 0:
            continue
        latest = float(state_data.iloc[-1]["arrivals"])
        items.append(StateSummary(state=s, latest_arrivals=latest))

    return SummaryResponse(as_of=as_of, states=items)


@app.post("/admin/clear-cache")
def clear_cache(x_admin_token: str | None = Header(default=None)) -> dict[str, str | int]:
    """Clear the in-memory model cache.

    Called by the Prefect retrain flow after promoting new models so the API
    serves the new versions on the next request instead of the cached old ones.

    Protected by a shared secret. The caller must send the ADMIN_TOKEN value
    in the 'X-Admin-Token' header. If ADMIN_TOKEN is not configured on the
    server, the endpoint is disabled (503) rather than left open.
    """
    expected = os.getenv("ADMIN_TOKEN")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Admin endpoint disabled: ADMIN_TOKEN is not configured.",
        )
    if x_admin_token != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing admin token.")

    from bundeshost.modeling.predict import _MODEL_CACHE

    n_before = len(_MODEL_CACHE)
    clear_model_cache()
    return {"status": "cleared", "models_evicted": n_before}
