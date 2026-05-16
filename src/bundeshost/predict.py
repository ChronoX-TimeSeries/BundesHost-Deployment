import joblib
import pandas as pd

from .config import MODEL_DIR, load_best_models
from .feature_engineering import create_corona_dummy

# ==================================================
# Load model for a given state
# ==================================================


def load_model(state_name):
    """
    Load the best fitted model for a given state, based on the
    selection saved in models/best_models.json.
    Returns (model, model_name).
    """

    best_models = load_best_models()

    if state_name not in best_models:
        raise KeyError(
            f"State '{state_name}' not found in best_models.json. "
            f"Run `python -m modeling.evaluate` to regenerate it."
        )

    best_type = best_models[state_name]["best_model"]
    model_path = MODEL_DIR / f"{state_name}_{best_type}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Run `python -m modeling.train` to regenerate models."
        )

    model = joblib.load(model_path)

    return model, model_path.name


# ==================================================
# Build future exogenous variable
# ==================================================


def build_future_exog(last_date, horizon):
    """
    Build a future COVID dummy aligned with the forecast horizon.
    """

    future_index = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )

    return create_corona_dummy(future_index)


# ==================================================
# Forecast function
# ==================================================


def forecast_state(state_name, horizon):
    """
    Generate a forecast for a state with 80% confidence intervals.
    Returns a DataFrame with columns: date, forecast, lower_ci, upper_ci.
    """

    model, model_name = load_model(state_name)

    # --------------------------------------------------
    # Determine last known date from the fitted model

    last_date = model.data.dates[-1]

    # --------------------------------------------------
    # Generate forecast

    if "sarimax" in model_name.lower():
        future_exog = build_future_exog(last_date, horizon)
        res = model.get_forecast(steps=horizon, exog=future_exog)
    else:
        res = model.get_forecast(steps=horizon)

    # --------------------------------------------------
    # Extract mean + 80% confidence interval

    mean = res.predicted_mean
    conf = res.conf_int(alpha=0.2)

    # --------------------------------------------------
    # Normalize index to Timestamp

    if isinstance(mean.index, pd.PeriodIndex):
        dates = mean.index.to_timestamp()
    else:
        dates = pd.to_datetime(mean.index)

    # --------------------------------------------------
    # Build final DataFrame

    forecast_df = pd.DataFrame(
        {
            "date": dates,
            "forecast": mean.values,
            "lower_ci": conf.iloc[:, 0].values,
            "upper_ci": conf.iloc[:, 1].values,
        }
    )

    return forecast_df
