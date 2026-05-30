"""
Modèles ML classiques : LinearRegression, RandomForest, XGBoost.
Chaque modèle est entraîné, évalué et sauvegardé automatiquement.
"""
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from src.utils.config import (
    LINEAR_MODEL_PATH, FOREST_MODEL_PATH, XGBOOST_MODEL_PATH,
    RF_PARAM_GRID, XGB_PARAMS, RANDOM_STATE,
)
from src.models.evaluator import compute_metrics


def train_linear_regression(X_train, y_train, X_test, y_test) -> dict:
    print("\n[LinearRegression] Entraînement...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, LINEAR_MODEL_PATH)
    metrics = compute_metrics(model.predict(X_test), y_test, "Linear Regression")
    return metrics


def train_random_forest(X_train, y_train, X_test, y_test) -> dict:
    print("\n[RandomForest] Recherche des hyperparamètres...")
    base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=RF_PARAM_GRID,
        n_iter=10,
        cv=3,
        scoring="r2",
        random_state=RANDOM_STATE,
        verbose=1,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print(f"[RandomForest] Meilleurs params : {search.best_params_}")
    joblib.dump(best_model, FOREST_MODEL_PATH)
    metrics = compute_metrics(best_model.predict(X_test), y_test, "Random Forest")
    return metrics


def train_xgboost(X_train, y_train, X_test, y_test) -> dict:
    print("\n[XGBoost] Entraînement...")
    model = XGBRegressor(**XGB_PARAMS, early_stopping_rounds=20, eval_metric="rmse")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    joblib.dump(model, XGBOOST_MODEL_PATH)
    metrics = compute_metrics(model.predict(X_test), y_test, "XGBoost")
    return metrics


def load_model(name: str):
    """Charge un modèle sauvegardé par son nom."""
    paths = {
        "linear":  LINEAR_MODEL_PATH,
        "forest":  FOREST_MODEL_PATH,
        "xgboost": XGBOOST_MODEL_PATH,
    }
    path = paths.get(name)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Modèle '{name}' introuvable. Lancez train.py d'abord.")
    return joblib.load(path)
