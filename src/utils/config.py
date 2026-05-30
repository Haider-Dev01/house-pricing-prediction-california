"""
Configuration centralisée du projet.
Tous les hyperparamètres et chemins sont définis ici.
"""
from pathlib import Path

# ── Chemins ─────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parents[2]
DATA_DIR   = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

DATA_PATH = DATA_DIR / "housing.csv"

# Chemins de sauvegarde des modèles
LINEAR_MODEL_PATH  = MODELS_DIR / "linear_regression.pkl"
FOREST_MODEL_PATH  = MODELS_DIR / "random_forest.pkl"
XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost.pkl"
DL_MODEL_PATH      = MODELS_DIR / "deep_learning.keras"
SCALER_PATH        = MODELS_DIR / "preprocessor.pkl"
METRICS_PATH       = MODELS_DIR / "metrics.json"

# ── Données ──────────────────────────────────────────────────────────────────
TARGET_COLUMN  = "median_house_value"
TEST_SIZE      = 0.2
RANDOM_STATE   = 42

NUMERIC_FEATURES = [
    "longitude", "latitude", "housing_median_age",
    "total_rooms", "total_bedrooms", "population",
    "households", "median_income",
]
CATEGORICAL_FEATURES = ["ocean_proximity"]

LOG_FEATURES = ["total_rooms", "total_bedrooms", "population", "households"]

# ── Hyperparamètres RandomForest ──────────────────────────────────────────────
RF_PARAM_GRID = {
    "regressor__n_estimators": [100, 200],
    "regressor__max_depth":    [None, 15, 30],
    "regressor__min_samples_split": [2, 5],
}

# ── Hyperparamètres XGBoost ───────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
}

# ── Hyperparamètres Deep Learning ─────────────────────────────────────────────
DL_CONFIG = {
    "epochs":          200,
    "batch_size":      256,
    "learning_rate":   1e-3,
    "dropout_rate":    0.3,
    "patience":        20,        # EarlyStopping
    "hidden_layers":   [256, 128, 64],
}

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ── Streamlit ─────────────────────────────────────────────────────────────────
APP_HOST = "0.0.0.0"
APP_PORT = 8501
