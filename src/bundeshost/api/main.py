"""FastAPI application entrypoint for the BundesHost API."""

from fastapi import FastAPI

from bundeshost.config import STATES

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