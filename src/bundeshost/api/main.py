"""FastAPI application entrypoint for the BundesHost API."""

from fastapi import FastAPI

app = FastAPI(
    title="BundesHost API",
    description="Tourism forecasting and hosting capacity analysis for German states.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}