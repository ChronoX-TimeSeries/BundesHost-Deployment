#import joblib

import statsmodels.api as sm

from .config import MODEL_DIR, MODEL_ORDERS
from .data_pipeline import get_tourism_data
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
# Train both models for one state
# ==================================================

def train_state(series, state, orders):
    """
    Train both SARIMA and SARIMAX for a single state.
    Returns a dict: {"sarima": fitted_model, "sarimax": fitted_model}.
    """

    results = {}

    # SARIMA
    order, seasonal_order = orders["sarima"]
    results["sarima"] = train_sarima(series, order, seasonal_order)

    # SARIMAX
    order, seasonal_order = orders["sarimax"]
    exog = create_corona_dummy(series.index)
    results["sarimax"] = train_sarimax(series, exog, order, seasonal_order)

    return results


# ==================================================
# Retrain all states (both SARIMA + SARIMAX)
# ==================================================

def retrain_all_states():
    """
    Train BOTH SARIMA and SARIMAX for every state and save them.
    Model selection happens later in evaluate.py.
    """

    print("START RETRAINING...\n")

    df = get_tourism_data()

    for state in MODEL_ORDERS.keys():

        print(f"Retraining {state}...")

        series = build_state_series(df, state)
        orders = MODEL_ORDERS[state]

        fitted_models = train_state(series, state, orders)

        for model_type, fitted in fitted_models.items():

            converged = (
                fitted.mle_retvals.get("converged", None)
                if hasattr(fitted, "mle_retvals") else None
            )

            model_path = MODEL_DIR / f"{state}_{model_type}.pkl"
            #joblib.dump(fitted, model_path)
            fitted.save(str(model_path), remove_data=True)
            print(f"  → {model_type:8s} converged={converged}  saved={model_path.name}")

        print()

    print("DONE.")


# ==================================================
# Run script
# ==================================================

if __name__ == "__main__":
    retrain_all_states()