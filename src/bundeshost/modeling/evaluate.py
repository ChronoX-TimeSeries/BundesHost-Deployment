import json

import numpy as np

from ..config import BEST_MODELS_PATH, MODEL_ORDERS
from ..data_pipeline import get_tourism_data
from .feature_engineering import build_state_series, create_corona_dummy
from .train import train_sarima, train_sarimax

# ==================================================
# Settings
# ==================================================

TEST_SIZE = 12  # last 12 months as test set


def load_best_models():
    """Load the saved best-model selection from JSON."""

    if not BEST_MODELS_PATH.exists():
        raise FileNotFoundError(
            f"Best models file not found at {BEST_MODELS_PATH}. "
            f"Run `python -m modeling.evaluate` first."
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
# Evaluate one state
# ==================================================


def evaluate_state(series, orders):
    """
    Train SARIMA and SARIMAX on the train portion,
    forecast the test portion, and return metrics for both.
    """

    train, test = train_test_split_ts(series)

    # --------------------------------------------------
    # SARIMA

    sarima_order, sarima_seasonal = orders["sarima"]
    sarima_model = train_sarima(train, sarima_order, sarima_seasonal)
    sarima_forecast = sarima_model.get_forecast(steps=TEST_SIZE).predicted_mean
    sarima_mape, sarima_mae = evaluate_forecast(test, sarima_forecast)

    # --------------------------------------------------
    # SARIMAX

    exog_train = create_corona_dummy(train.index)
    exog_test = create_corona_dummy(test.index)

    sarimax_order, sarimax_seasonal = orders["sarimax"]
    sarimax_model = train_sarimax(train, exog_train, sarimax_order, sarimax_seasonal)
    sarimax_forecast = sarimax_model.get_forecast(steps=TEST_SIZE, exog=exog_test).predicted_mean
    sarimax_mape, sarimax_mae = evaluate_forecast(test, sarimax_forecast)

    return {
        "sarima": {"mape": sarima_mape, "mae": sarima_mae},
        "sarimax": {"mape": sarimax_mape, "mae": sarimax_mae},
    }


# ==================================================
# Evaluate all states and select best
# ==================================================


def evaluate_all_states():
    """
    Evaluate SARIMA and SARIMAX for every state on the test set,
    select the best (lowest MAPE), and save the selection to JSON.
    """

    print("START EVALUATION...\n")

    df = get_tourism_data()

    results = {}

    for state in MODEL_ORDERS.keys():

        print(f"Evaluating {state}...")

        series = build_state_series(df, state)
        orders = MODEL_ORDERS[state]

        metrics = evaluate_state(series, orders)

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

    return results


# ==================================================
# Run script
# ==================================================

if __name__ == "__main__":
    evaluate_all_states()
