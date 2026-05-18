"""FastAPI application entrypoint for the BundesHost API."""

from datetime import date

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from bundeshost.api.schemas import ForecastPoint, ForecastResponse
from bundeshost.config import STATES
from bundeshost.modeling.predict import forecast_state, get_last_training_date

app = FastAPI(
    title="BundesHost API",
    description="Tourism forecasting and hosting capacity analysis for German states.",
    version="0.1.0",
)


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

    # Tag each point as 'backfill' or 'future' based on its date
    points = [
        ForecastPoint(
            date=row["date"].date(),
            forecast=float(row["forecast"]),
            lower_ci=float(row["lower_ci"]),
            upper_ci=float(row["upper_ci"]),
            type="backfill" if row["date"] < today else "future",
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