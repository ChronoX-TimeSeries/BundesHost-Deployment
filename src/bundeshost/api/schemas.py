"""Pydantic schemas for the BundesHost API."""

from datetime import date
from typing import Literal

from pydantic import BaseModel


class ForecastPoint(BaseModel):
    """A single point in a forecast time series."""

    date: date
    forecast: float
    lower_ci: float
    upper_ci: float
    type: Literal["backfill", "future"]


class ForecastResponse(BaseModel):
    """Forecast response for a single state."""

    state: str
    horizon: int
    today: date
    last_training_date: date
    forecast: list[ForecastPoint]