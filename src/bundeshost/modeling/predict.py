import mlflow.statsmodels
import pandas as pd
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

from .feature_engineering import create_corona_dummy

# ==================================================
# In-memory model cache
# ==================================================
# Models are loaded from the MLflow Model Registry on first access and kept
# in memory. This avoids repeated network round-trips when the API serves
# multiple forecasts.

_MODEL_CACHE: dict[str, tuple] = {}


# ==================================================
# Load model for a given state (cached, from MLflow Registry)
# ==================================================


def load_model(state_name):
    """
    Load the Production version of the model for a given state from the
    MLflow Model Registry. Caches the loaded model in memory so subsequent
    calls are free.

    Returns (model, model_name).

    The returned model_name string includes the variant (e.g.
    "Hamburg_sarima") so that forecast_state() can detect SARIMAX models
    via substring match on "sarimax".

    Raises
    ------
    KeyError
        If no registered model exists for the given state.
    FileNotFoundError
        If the registered model exists but has no Production version.
    """

    if state_name in _MODEL_CACHE:
        return _MODEL_CACHE[state_name]

    registered_name = f"bundeshost-{state_name}"
    model_uri = f"models:/{registered_name}@production"

    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(registered_name, "production")
    except RestException as e:
        msg = str(e)
        if (
            "RESOURCE_DOES_NOT_EXIST" in msg
            or "INVALID_PARAMETER_VALUE" in msg
            or "not found" in msg.lower()
        ):
            raise FileNotFoundError(
                f"No '@production' model found for {registered_name}. "
                f"Run `python scripts/promote_all_to_production.py` to seed it, "
                f"or `python -m bundeshost.modeling.train` if it isn't registered yet."
            ) from e
        raise

    variant = version.tags.get("variant", "sarima")
    model = mlflow.statsmodels.load_model(model_uri)

    model_name = f"{state_name}_{variant}"
    _MODEL_CACHE[state_name] = (model, model_name)

    return model, model_name


# ==================================================
# Last training date helper
# ==================================================


def get_last_training_date(state_name) -> pd.Timestamp:
    """
    Return the last date the model was trained on.
    Used by the API to compute the gap between training data and 'today'.
    """

    model, _ = load_model(state_name)

    return pd.Timestamp(model.data.dates[-1])


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
