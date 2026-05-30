"""
Pipeline de prétraitement sklearn unifié.
Élimine tout risque de data leakage grâce au ColumnTransformer.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib

from src.utils.config import (
    TARGET_COLUMN, TEST_SIZE, RANDOM_STATE,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, LOG_FEATURES,
    SCALER_PATH,
)


# ── Feature engineering ───────────────────────────────────────────────────────

def _add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des features dérivées (doit recevoir un DataFrame)."""
    X = X.copy()
    X["bedroom_ratio"]    = X["total_bedrooms"] / (X["total_rooms"] + 1)
    X["household_rooms"]  = X["total_rooms"]    / (X["households"]  + 1)
    X["income_per_room"]  = X["median_income"]  / (X["total_rooms"] + 1)
    return X


def _log_transform(X: pd.DataFrame) -> pd.DataFrame:
    """Applique log1p sur les colonnes skewed."""
    X = X.copy()
    for col in LOG_FEATURES:
        if col in X.columns:
            X[col] = np.log1p(X[col])
    return X


# ── Colonnes après feature engineering ───────────────────────────────────────

NUMERIC_ALL = NUMERIC_FEATURES + ["bedroom_ratio", "household_rooms", "income_per_room"]


def build_preprocessor() -> ColumnTransformer:
    """
    Construit le ColumnTransformer complet :
      - Numérique : Imputer → log1p → StandardScaler
      - Catégoriel : Imputer → OneHotEncoder
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline,  NUMERIC_ALL),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def prepare_data(df: pd.DataFrame):
    """
    Applique le feature engineering et sépare X / y.
    Returns: X_train, X_test, y_train, y_test, preprocessor (fitted)
    """
    # Feature engineering (avant split pour éviter les NaN sur les ratios)
    df = _add_features(df)
    df = _log_transform(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p  = preprocessor.transform(X_test)

    # Sauvegarde du préprocesseur
    joblib.dump(preprocessor, SCALER_PATH)
    print(f"[Preprocessor] Sauvegardé → {SCALER_PATH}")

    return X_train_p, X_test_p, y_train.values, y_test.values, preprocessor


def preprocess_single(data: dict, preprocessor) -> np.ndarray:
    """Prépare une seule observation pour l'inférence."""
    df = pd.DataFrame([data])
    df = _add_features(df)
    df = _log_transform(df)
    return preprocessor.transform(df)
