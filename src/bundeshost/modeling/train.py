import mlflow
import mlflow.statsmodels
import statsmodels.api as sm
from mlflow.tracking import MlflowClient

from ..config import MODEL_ORDERS, load_best_models
from ..data.pipeline import get_tourism_data
from ..registry import promote_to_production, should_promote
from .feature_engineering import build_state_series, create_corona_dummy

EXPERIMENT_NAME = "bundeshost-training"


# ==================================================
# Train a single SARIMA model
# ==================================================


def train_sarima(series, order, seasonal_order):
    """
    Fit a SARIMA model on the full series.
    """

    model = sm.tsa.arima.ARIMA(
        series,
        order=order,
        seasonal_order=seasonal_order,
    ).fit(method_kwargs={"maxiter": 500})

    return model


# ==================================================
# Train a single SARIMAX model with COVID dummy
# ==================================================


def train_sarimax(series, exog, order, seasonal_order):
    """
    Fit a SARIMAX model on the full series with COVID dummy as exog.
    """

    model = sm.tsa.statespace.SARIMAX(
        series,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
    ).fit(disp=False, maxiter=500)

    return model


# ==================================================
# Train best model for one state (based on best_models.json)
# ==================================================


def train_best_for_state(series, state, orders, best_type):
    """
    Train ONLY the best model (SARIMA or SARIMAX) for a single state
    on the full data.
    """

    order, seasonal_order = orders[best_type]

    if best_type == "sarimax":
        exog = create_corona_dummy(series.index)
        fitted = train_sarimax(series, exog, order, seasonal_order)
    else:
        fitted = train_sarima(series, order, seasonal_order)

    return fitted


# ==================================================
# Retrain best model for ONE state (with MLflow logging + registry + auto-promotion)
# ==================================================


def retrain_state(state, df=None, best_models=None, order=None, seasonal_order=None):
    """
    Retrain the best model for a single state on the full data, log the run
    to MLflow, register the model in the MLflow Model Registry, and (if it
    beats the current Production MAPE) promote it to Production.

    Parameters
    ----------
    state : str
        The state name (e.g. "Hamburg"). Must exist in MODEL_ORDERS.
    df : pd.DataFrame, optional
        The tourism dataframe. If None, loaded from the pipeline.
    best_models : dict, optional
        The best_models.json contents. If None, loaded from disk.
    order : tuple, optional
        Override (p, d, q) from MODEL_ORDERS. Useful for ad-hoc experiments.
    seasonal_order : tuple, optional
        Override (P, D, Q, s) from MODEL_ORDERS.

    Returns
    -------
    ModelVersion
        The registered model version object.

    Notes
    -----
    If there is no active MLflow run when this is called, a new top-level run
    is created. If there is one (e.g. from retrain_all_states), the run is
    created as a nested child.
    """
    if df is None:
        df = get_tourism_data()
    if best_models is None:
        best_models = load_best_models()

    if state not in best_models:
        raise ValueError(f"{state!r} is not in best_models.json")

    best_type = best_models[state]["best_model"]
    default_order, default_seasonal = MODEL_ORDERS[state][best_type]
    order = order or default_order
    seasonal_order = seasonal_order or default_seasonal

    # MAPE from the latest evaluation — used as the promotion criterion.
    new_mape = float(best_models[state]["metrics"][best_type]["mape"])

    # Smart-detect: if a parent run is active, become a child; otherwise stand alone.
    is_nested = mlflow.active_run() is not None

    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    print(f"Retraining {state} ({best_type})...")

    with mlflow.start_run(run_name=state, nested=is_nested) as child_run:
        mlflow.log_param("state", state)
        mlflow.log_param("model_type", best_type)
        mlflow.log_param("order", str(order))
        mlflow.log_param("seasonal_order", str(seasonal_order))
        mlflow.log_param("eval_mape", new_mape)

        series = build_state_series(df, state)

        # Use overridden orders if provided, otherwise the defaults
        orders = {best_type: (order, seasonal_order)}
        fitted = train_best_for_state(series, state, orders, best_type)

        converged = (
            fitted.mle_retvals.get("converged", None)
            if hasattr(fitted, "mle_retvals")
            else None
        )

        mlflow.log_metric("aic", float(fitted.aic))
        mlflow.log_metric("bic", float(fitted.bic))
        mlflow.log_metric("converged", 1.0 if converged else 0.0)

        mlflow.statsmodels.log_model(fitted, artifact_path="model")

        # Register in the MLflow Model Registry
        model_uri = f"runs:/{child_run.info.run_id}/model"
        registered_name = f"bundeshost-{state}"
        model_version = mlflow.register_model(model_uri, registered_name)

        # Tag with the variant (variant is not part of the name)
        client.set_model_version_tag(
            name=registered_name,
            version=model_version.version,
            key="variant",
            value=best_type,
        )

        # Auto-promotion: if this version beats current Production MAPE, promote it.
        if should_promote(client, registered_name, new_mape):
            promote_to_production(client, registered_name, model_version.version, new_mape)
            promotion_note = f"promoted to Production (MAPE={new_mape:.2f}%)"
        else:
            promotion_note = f"not promoted (MAPE {new_mape:.2f}% >= current)"

        print(
            f"  → converged={converged}  "
            f"registered={registered_name} v{model_version.version} "
            f"(variant={best_type})  "
            f"{promotion_note}\n"
        )

    return model_version


# ==================================================
# Retrain best model for all states (with MLflow logging + registry)
# ==================================================


def retrain_all_states():
    """
    For every state, train the best model (per best_models.json) on the
    full data, log the run to MLflow, and register the model in the
    MLflow Model Registry.

    Prerequisite: best_models.json must exist (run `python -m bundeshost.modeling.evaluate` first).
    """

    print("START RETRAINING (best models only)...\n")

    mlflow.set_experiment(EXPERIMENT_NAME)

    best_models = load_best_models()
    df = get_tourism_data()

    with mlflow.start_run(run_name="retrain-all") as parent_run:
        mlflow.log_param("total_states", len(MODEL_ORDERS))
        mlflow.log_param("data_source", "postgres-mart")

        for state in MODEL_ORDERS.keys():
            if state not in best_models:
                print(f"  ⚠️  Skipping {state} (not in best_models.json)")
                continue

            retrain_state(state, df=df, best_models=best_models)

        print(f"DONE. Parent run id: {parent_run.info.run_id}")


# ==================================================
# Run script
# ==================================================

if __name__ == "__main__":
    retrain_all_states()
