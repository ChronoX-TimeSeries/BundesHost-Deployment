import json

import mlflow
import numpy as np

from ..config import BEST_MODELS_PATH, MODEL_ORDERS
from ..data.pipeline import get_tourism_data
from .feature_engineering import build_state_series, create_corona_dummy
from .train import train_sarima, train_sarimax

# ==================================================
# Settings
# ==================================================

TEST_SIZE = 12  # last 12 months as test set
EXPERIMENT_NAME = "bundeshost-evaluation"


def load_best_models():
    """Load the saved best-model selection from JSON."""

    if not BEST_MODELS_PATH.exists():
        raise FileNotFoundError(
            f"Best models file not found at {BEST_MODELS_PATH}. "
            f"Run `python -m bundeshost.modeling.evaluate` first."
        )

    return json.loads(BEST_MODELS_PATH.read_text())


# ==================================================
# Train / test split
# ==================================================


def train_test_split_ts(series, test_size=TEST_SIZE):
    """Split a time series into train (all but last N) and test (last N)."""
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    return train, test


# ==================================================
# Evaluation metrics
# ==================================================


def evaluate_forecast(y_true, y_pred):
    """Compute MAPE (%) and MAE between actual and predicted series."""

    y_true, y_pred = y_true.align(y_pred, join="inner")
    y_true = y_true.values
    y_pred = y_pred.values

    # Avoid division by zero in MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    mae = np.mean(np.abs(y_true - y_pred))

    return float(mape), float(mae)


# ==================================================
# Evaluate one variant for one state (with MLflow logging)
# ==================================================


def _evaluate_variant(state, variant, train, test, orders):
    """Train one variant on train, forecast test, log to MLflow, return metrics."""

    order, seasonal_order = orders[variant]

    with mlflow.start_run(run_name=f"{state}-{variant}", nested=True):
        mlflow.log_param("state", state)
        mlflow.log_param("model_type", variant)
        mlflow.log_param("order", str(order))
        mlflow.log_param("seasonal_order", str(seasonal_order))
        mlflow.log_param("test_start", str(test.index[0].date()))
        mlflow.log_param("test_end", str(test.index[-1].date()))
        mlflow.log_param("train_size", len(train))
        mlflow.log_param("test_size", len(test))

        if variant == "sarimax":
            exog_train = create_corona_dummy(train.index)
            exog_test = create_corona_dummy(test.index)
            model = train_sarimax(train, exog_train, order, seasonal_order)
            forecast = model.get_forecast(steps=len(test), exog=exog_test).predicted_mean
        else:
            model = train_sarima(train, order, seasonal_order)
            forecast = model.get_forecast(steps=len(test)).predicted_mean

        mape, mae = evaluate_forecast(test, forecast)

        mlflow.log_metric("mape", mape)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("aic", float(model.aic))
        mlflow.log_metric("bic", float(model.bic))

    return {"mape": mape, "mae": mae}


# ==================================================
# Evaluate one state (both variants)
# ==================================================


def evaluate_state(state, series, orders):
    """
    Train SARIMA and SARIMAX on the train portion,
    forecast the test portion, log each to MLflow, and return metrics for both.
    """

    train, test = train_test_split_ts(series)

    return {
        "sarima": _evaluate_variant(state, "sarima", train, test, orders),
        "sarimax": _evaluate_variant(state, "sarimax", train, test, orders),
    }


# ==================================================
# Evaluate all states and select best
# ==================================================


def evaluate_all_states():
    """
    Evaluate SARIMA and SARIMAX for every state on the test set,
    select the best (lowest MAPE), log all 32 runs to MLflow,
    and save the selection to JSON.
    """

    print("START EVALUATION...\n")

    mlflow.set_experiment(EXPERIMENT_NAME)

    df = get_tourism_data()

    results = {}

    with mlflow.start_run(run_name="evaluate-all") as parent_run:
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("total_states", len(MODEL_ORDERS))
        mlflow.log_param("data_source", "postgres-mart")

        for state in MODEL_ORDERS.keys():

            print(f"Evaluating {state}...")

            series = build_state_series(df, state)
            orders = MODEL_ORDERS[state]

            metrics = evaluate_state(state, series, orders)

            # Pick winner (lowest MAPE)
            best_model = min(metrics, key=lambda m: metrics[m]["mape"])

            results[state] = {
                "best_model": best_model,
                "metrics": metrics,
            }

            for m, vals in metrics.items():
                marker = "  ★" if m == best_model else "   "
                print(f"  {marker} {m:8s} MAPE={vals['mape']:6.2f}%  MAE={vals['mae']:>10,.0f}")
            print()

        # --------------------------------------------------
        # Save selection to JSON

        BEST_MODELS_PATH.write_text(json.dumps(results, indent=2))
        print(f"DONE. Saved → {BEST_MODELS_PATH}")
        print(f"Parent run id: {parent_run.info.run_id}")

    return results


# ==================================================
# Run script
# ==================================================

if __name__ == "__main__":
    evaluate_all_states()
