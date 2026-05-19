"""Helpers for interacting with the MLflow Model Registry.

Single source of truth for promotion logic, version lookups, and the
in-memory model cache. Used by:
- modeling.train: to decide whether to promote a newly trained version
- modeling.predict: holds the in-process model cache
- Phase 2E (Prefect retrain flow): same promotion logic for scheduled retrains
- API admin endpoints: to clear the cache after a retrain
"""

from typing import Optional

from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient


# ==================================================
# Lookup helpers
# ==================================================


def get_production_version(client: MlflowClient, registered_name: str) -> Optional[ModelVersion]:
    """Return the current Production version of a registered model, or None.

    Returns None if the model is not registered yet, or if it has no
    Production version.
    """
    try:
        versions = client.get_latest_versions(registered_name, stages=["Production"])
    except RestException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e):
            return None
        raise
    return versions[0] if versions else None


def get_production_mape(client: MlflowClient, registered_name: str) -> Optional[float]:
    """Return the MAPE that was tagged on the current Production version, or None."""
    version = get_production_version(client, registered_name)
    if version is None:
        return None
    raw = version.tags.get("production_mape")
    return float(raw) if raw is not None else None


# ==================================================
# Promotion logic
# ==================================================


def should_promote(client: MlflowClient, registered_name: str, new_mape: float) -> bool:
    """Decide whether a new version should be promoted to Production.

    Promote if either:
      - There is no current Production version, or
      - The new MAPE is strictly better than the current Production MAPE.

    If the current Production version has no production_mape tag (e.g.
    seeded manually before this step), we promote — the tag-based history
    starts fresh from there.
    """
    current_mape = get_production_mape(client, registered_name)
    if current_mape is None:
        return True
    return new_mape < current_mape


def promote_to_production(
    client: MlflowClient,
    registered_name: str,
    version: str,
    mape: float,
) -> None:
    """Promote a version to Production and tag it with its MAPE.

    Archives any previous Production version so that get_latest_versions
    keeps returning a single Production entry.
    """
    client.transition_model_version_stage(
        name=registered_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    client.set_model_version_tag(
        name=registered_name,
        version=version,
        key="production_mape",
        value=str(mape),
    )


# ==================================================
# Cache management
# ==================================================


def clear_model_cache() -> None:
    """Clear the in-process model cache held by modeling.predict.

    Local-import to avoid a circular import (predict imports nothing from
    registry at module load).
    """
    from .modeling import predict

    predict._MODEL_CACHE.clear()
