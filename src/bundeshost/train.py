import joblib

import statsmodels.api as sm

from .config import MODEL_DIR, MODEL_ORDERS
from .data_pipeline import get_tourism_data
from .config import load_best_models
from .feature_engineering import build_state_series, create_corona_dummy


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
# Retrain best model for all states
# ==================================================

def retrain_all_states():
    """
    For every state, train the best model (per best_models.json) on the
    full data and save it as a .pkl file.

    Prerequisite: best_models.json must exist (run `python -m modeling.evaluate` first).
    """

    print("START RETRAINING (best models only)...\n")

    best_models = load_best_models()
    df = get_tourism_data()

    for state in MODEL_ORDERS.keys():

        if state not in best_models:
            print(f"  ⚠️  Skipping {state} (not in best_models.json)")
            continue

        best_type = best_models[state]["best_model"]

        print(f"Retraining {state} ({best_type})...")

        series = build_state_series(df, state)
        orders = MODEL_ORDERS[state]

        fitted = train_best_for_state(series, state, orders, best_type)

        converged = (
            fitted.mle_retvals.get("converged", None)
            if hasattr(fitted, "mle_retvals") else None
        )

        model_path = MODEL_DIR / f"{state}_{best_type}.pkl"
        joblib.dump(fitted, model_path)

        print(f"  → converged={converged}  saved={model_path.name}\n")

    print("DONE.")


# ==================================================
# Run script
# ==================================================

if __name__ == "__main__":
    retrain_all_states()