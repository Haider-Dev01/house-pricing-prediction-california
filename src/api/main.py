"""
FastAPI — API de prédiction des prix immobiliers.
Endpoints :
  POST /predict          → prédiction unique
  POST /predict/batch    → prédictions multiples
  GET  /models/compare   → scores de tous les modèles
  GET  /health           → healthcheck
Swagger UI disponible sur http://localhost:8000/docs
"""
import sys
import json
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data.preprocessor import preprocess_single
from src.models.classical import load_model
from src.models.deep_learning import load_dl_model
from src.models.evaluator import load_all_metrics
from src.utils.config import SCALER_PATH, METRICS_PATH

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="California House Price Prediction API",
    description="API ML/DL pour prédire les prix immobiliers en Californie.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Chargement des modèles au démarrage ───────────────────────────────────────
@app.on_event("startup")
async def load_models():
    global _preprocessor, _models
    if not SCALER_PATH.exists():
        print("⚠️  Aucun modèle trouvé — lancez train.py d'abord.")
        _preprocessor = None
        _models = {}
        return

    _preprocessor = joblib.load(SCALER_PATH)
    _models = {}
    for name in ("linear", "forest", "xgboost"):
        try:
            _models[name] = load_model(name)
        except FileNotFoundError:
            pass
    try:
        _models["deep_learning"] = load_dl_model()
    except FileNotFoundError:
        pass
    print(f"[API] Modèles chargés : {list(_models.keys())}")


# ── Schémas Pydantic ──────────────────────────────────────────────────────────
class HouseFeatures(BaseModel):
    longitude:           float = Field(..., example=-122.23)
    latitude:            float = Field(..., example=37.88)
    housing_median_age:  float = Field(..., example=41.0)
    total_rooms:         float = Field(..., example=880.0)
    total_bedrooms:      float = Field(..., example=129.0)
    population:          float = Field(..., example=322.0)
    households:          float = Field(..., example=126.0)
    median_income:       float = Field(..., example=8.3252)
    ocean_proximity:     str   = Field(..., example="NEAR BAY")


ModelName = Literal["linear", "forest", "xgboost", "deep_learning"]


class PredictionResponse(BaseModel):
    model:      str
    prediction: float
    unit:       str = "USD"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(_models.keys())}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures, model_name: ModelName = "xgboost"):
    """Prédit le prix médian d'un logement californien."""
    if not _models:
        raise HTTPException(503, "Modèles non chargés — lancez train.py d'abord.")
    if model_name not in _models:
        raise HTTPException(404, f"Modèle '{model_name}' non disponible.")

    X = preprocess_single(features.model_dump(), _preprocessor)
    model = _models[model_name]

    if model_name == "deep_learning":
        pred = float(model.predict(X, verbose=0).flatten()[0])
    else:
        pred = float(model.predict(X)[0])

    return PredictionResponse(model=model_name, prediction=round(pred, 2))


@app.post("/predict/batch")
def predict_batch(features_list: list[HouseFeatures], model_name: ModelName = "xgboost"):
    """Prédictions sur un lot d'observations."""
    return [
        predict(f, model_name) for f in features_list
    ]


@app.get("/models/compare")
def compare_models():
    """Retourne les métriques de tous les modèles entraînés."""
    metrics = load_all_metrics()
    if not metrics:
        raise HTTPException(404, "Aucune métrique trouvée — lancez train.py d'abord.")
    return {"models": sorted(metrics, key=lambda m: m["R2"], reverse=True)}
