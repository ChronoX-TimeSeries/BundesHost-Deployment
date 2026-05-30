"""Seed the '@production' alias for every bundeshost-* model.

One-off / occasional script: points each registered model's 'production'
alias at its latest version. Run it once after migrating from stages to
aliases (so predict.py can resolve models:/bundeshost-{state}@production),
or any time a model is missing its production alias.

Normal retrains handle promotion automatically via registry.promote_to_production.
"""

import sys

from mlflow.tracking import MlflowClient

from bundeshost.config import MODEL_ORDERS
from bundeshost.registry import PRODUCTION_ALIAS


def main() -> int:
    client = MlflowClient()
    promoted = 0
    skipped = 0

    for state in MODEL_ORDERS.keys():
        registered_name = f"bundeshost-{state}"

        try:
            versions = client.search_model_versions(f"name='{registered_name}'")
        except Exception as e:
            print(f"  x {registered_name}: {e}")
            skipped += 1
            continue

        if not versions:
            print(f"  x {registered_name}: no versions found")
            skipped += 1
            continue

        # Prefer the version that still carries the legacy 'Production' stage
        # (that is the real production model). Fall back to the latest version
        # only if no version was ever staged Production.
        staged = [v for v in versions if v.current_stage == "Production"]
        if staged:
            target = max(staged, key=lambda v: int(v.version))
        else:
            target = max(versions, key=lambda v: int(v.version))

        # Is the alias already on the target version?
        try:
            current = client.get_model_version_by_alias(registered_name, PRODUCTION_ALIAS)
            if current.version == target.version:
                print(f"  - {registered_name} v{target.version} already @{PRODUCTION_ALIAS}")
                continue
        except Exception:
            pass  # alias not set yet; fall through and set it

        client.set_registered_model_alias(
            name=registered_name,
            alias=PRODUCTION_ALIAS,
            version=target.version,
        )
        print(f"  v {registered_name} v{target.version} -> @{PRODUCTION_ALIAS}")
        promoted += 1

    print(f"\nDONE. Aliased: {promoted}, Skipped: {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
