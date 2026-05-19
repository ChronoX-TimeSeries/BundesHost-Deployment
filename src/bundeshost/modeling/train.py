import joblib
import mlflow
import mlflow.statsmodels
import statsmodels.api as sm

from ..config import MODEL_DIR, MODEL_ORDERS, load_best_models
from ..data.pipeline import get_tourism_data
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
# Retrain best model for all states (with MLflow logging)
# ==================================================


def retrain_all_states():
    """
    For every state, train the best model (per best_models.json) on the
    full data, save it as a .pkl file, and log the run to MLflow.

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

            best_type = best_models[state]["best_model"]
            order, seasonal_order = MODEL_ORDERS[state][best_type]

            print(f"Retraining {state} ({best_type})...")

            with mlflow.start_run(run_name=state, nested=True):
                mlflow.log_param("state", state)
                mlflow.log_param("model_type", best_type)
                mlflow.log_param("order", str(order))
                mlflow.log_param("seasonal_order", str(seasonal_order))

                series = build_state_series(df, state)
                orders = MODEL_ORDERS[state]

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

                model_path = MODEL_DIR / f"{state}_{best_type}.pkl"
                joblib.dump(fitted, model_path)

                print(f"  → converged={converged}  saved={model_path.name}\n")

        print(f"DONE. Parent run id: {parent_run.info.run_id}")


# ==================================================
# Run script
# ==================================================

if __name__ == "__main__":
    retrain_all_states()
