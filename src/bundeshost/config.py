from pathlib import Path


# ==================================================
# Paths
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

# DATA_PATH = BASE_DIR / "data" / "processed" / "tourism_long.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ==================================================
# Forecast settings
# ==================================================

DEFAULT_FORECAST_HORIZON = 12


# ==================================================
# Dataset states
# ==================================================

STATES = [
    "Baden-Württemberg",
    "Bayern",
    "Berlin",
    "Brandenburg",
    "Bremen",
    "Hamburg",
    "Hessen",
    "Mecklenburg-Vorpommern",
    "Niedersachsen",
    "Nordrhein-Westfalen",
    "Rheinland-Pfalz",
    "Saarland",
    "Sachsen",
    "Sachsen-Anhalt",
    "Schleswig-Holstein",
    "Thüringen",
]


# ==================================================
# COVID dummy period
# ==================================================

COVID_START = "2020-03-01"
COVID_END = "2021-05-01"


# ==================================================
# SARIMA / SARIMAX orders per state
# ==================================================

MODEL_ORDERS = {

    "Baden-Württemberg": {
        "sarima": ((2, 1, 2), (1, 1, 1, 12)),
        "sarimax": ((2, 1, 2), (1, 1, 1, 12)),
    },

    "Bayern": {
        "sarima": ((2, 1, 2), (1, 1, 1, 12)),
        "sarimax": ((1, 1, 2), (1, 1, 1, 12)),
    },

    "Berlin": {
        "sarima": ((2, 1, 2), (1, 1, 1, 12)),
        "sarimax": ((2, 1, 2), (1, 1, 1, 12)),
    },

    "Brandenburg": {
        "sarima": ((4, 1, 4), (1, 1, 1, 12)),
        "sarimax": ((3, 1, 3), (1, 1, 1, 12)),
    },

    "Bremen": {
        "sarima": ((3, 1, 2), (0, 1, 1, 12)),
        "sarimax": ((2, 1, 2), (0, 1, 1, 12)),
    },

    "Hamburg": {
        "sarima": ((2, 1, 2), (0, 1, 1, 12)),
        "sarimax": ((1, 1, 2), (0, 1, 1, 12)),
    },

    "Hessen": {
        "sarima": ((2, 1, 3), (0, 1, 1, 12)),
        "sarimax": ((2, 1, 3), (0, 1, 1, 12)),
    },

    "Mecklenburg-Vorpommern": {
        "sarima": ((1, 1, 1), (1, 1, 1, 12)),
        "sarimax": ((1, 1, 1), (1, 1, 1, 12)),
    },

    "Niedersachsen": {
        "sarima": ((4, 1, 3), (0, 1, 1, 12)),
        "sarimax": ((4, 1, 3), (0, 1, 1, 12)),
    },

    "Nordrhein-Westfalen": {
        "sarima": ((1, 1, 1), (1, 1, 1, 12)),
        "sarimax": ((1, 1, 1), (1, 1, 1, 12)),
    },

    "Rheinland-Pfalz": {
        "sarima": ((2, 1, 2), (1, 1, 1, 12)),
        "sarimax": ((2, 1, 1), (1, 1, 1, 12)),
    },

    "Saarland": {
        "sarima": ((3, 1, 3), (0, 1, 1, 12)),
        "sarimax": ((1, 1, 1), (1, 1, 1, 12)),
    },

    "Sachsen": {
        "sarima": ((2, 1, 2), (1, 1, 1, 12)),
        "sarimax": ((2, 1, 2), (1, 1, 1, 12)),
    },

    "Sachsen-Anhalt": {
        "sarima": ((1, 1, 2), (1, 1, 1, 12)),
        "sarimax": ((2, 1, 1), (1, 1, 1, 12)),
    },

    "Schleswig-Holstein": {
        "sarima": ((1, 1, 2), (1, 1, 1, 12)),
        "sarimax": ((2, 1, 2), (0, 1, 1, 12)),
    },

    "Thüringen": {
        "sarima": ((1, 1, 2), (1, 1, 1, 12)),
        "sarimax": ((1, 1, 2), (1, 1, 1, 12)),
    },
}
import json


# ==================================================
# Best models JSON file
# ==================================================

BEST_MODELS_PATH = MODEL_DIR / "best_models.json"


def load_best_models():
    """Load the saved best-model selection from JSON."""

    if not BEST_MODELS_PATH.exists():
        raise FileNotFoundError(
            f"Best models file not found at {BEST_MODELS_PATH}. "
            f"Run `python -m modeling.evaluate` first."
        )

    return json.loads(BEST_MODELS_PATH.read_text())
