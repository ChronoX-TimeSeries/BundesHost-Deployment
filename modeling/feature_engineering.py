import pandas as pd

from .config import COVID_START, COVID_END


# ==================================================
# Build univariate state time series
# ==================================================

def build_state_series(df, state, target="arrivals"):
    """
    Build a monthly univariate time series for a single state.
    """

    df_state = df[df["state"] == state].copy()
    df_state = df_state.set_index("date")

    series = pd.Series(
        df_state[target].values,
        index=df_state.index,
        name=target,
    )

    # Ensure monthly frequency (start of month)
    series = series.asfreq("MS")

    return series


# ==================================================
# COVID dummy for SARIMAX
# ==================================================

def create_corona_dummy(index):
    """
    Create COVID-19 dummy variable for SARIMAX models.
    Returns 1 during the COVID period, 0 elsewhere.
    """

    dummy = pd.Series(
        (
            (index >= pd.Timestamp(COVID_START))
            & (index <= pd.Timestamp(COVID_END))
        ).astype(int),
        index=index,
        name="corona_dummy",
    )

    return dummy