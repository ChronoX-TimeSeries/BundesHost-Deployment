"""Pydantic schemas for the BundesHost API."""

from datetime import date
from typing import Literal

from pydantic import BaseModel


# ==================================================
# Forecast schemas
# ==================================================


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


# ==================================================
# History schemas
# ==================================================


class HistoryPoint(BaseModel):
    """A single point in a historical time series."""

    date: date
    arrivals: float


class HistoryResponse(BaseModel):
    """Historical monthly arrivals for a single state."""

    state: str
    history: list[HistoryPoint]


# ==================================================
# Summary schemas
# ==================================================


class StateSummary(BaseModel):
    """Latest available arrivals for one state."""

    state: str
    latest_arrivals: float


class SummaryResponse(BaseModel):
    """Snapshot of the latest arrivals across all 16 states."""

    as_of: date
    states: list[StateSummary]