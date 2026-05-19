"""Promote the latest version of every bundeshost-* model to Production.

One-off script run after Step 4 to seed the Production stage so that
predict.py can resolve models:/bundeshost-{state}/Production.

After Step 6 ships, retrain_state() will handle promotion automatically.
"""

import sys

from mlflow.tracking import MlflowClient

from bundeshost.config import MODEL_ORDERS


def main() -> int:
    client = MlflowClient()
    promoted = 0
    skipped = 0

    for state in MODEL_ORDERS.keys():
        registered_name = f"bundeshost-{state}"

        try:
            versions = client.search_model_versions(f"name='{registered_name}'")
        except Exception as e:
            print(f"  ✗ {registered_name}: {e}")
            skipped += 1
            continue

        if not versions:
            print(f"  ✗ {registered_name}: no versions found")
            skipped += 1
            continue

        # Pick the latest version (highest version number)
        latest = max(versions, key=lambda v: int(v.version))

        if latest.current_stage == "Production":
            print(f"  - {registered_name} v{latest.version} already Production")
            continue

        client.transition_model_version_stage(
            name=registered_name,
            version=latest.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"  ✓ {registered_name} v{latest.version} → Production")
        promoted += 1

    print(f"\nDONE. Promoted: {promoted}, Skipped: {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
