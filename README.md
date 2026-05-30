# California House Price Prediction 🏡

Projet ML/DL complet de prédiction des prix immobiliers en Californie,
avec une API FastAPI et une interface Streamlit moderne.

## 🏗️ Architecture

```
house-pricing-prediction-california/
├── data/                        # Dataset (housing.csv)
├── models/                      # Modèles sauvegardés (auto-générés)
├── src/
│   ├── data/
│   │   ├── loader.py            # Chargement & validation
│   │   └── preprocessor.py     # Pipeline sklearn unifié (no leakage)
│   ├── models/
│   │   ├── classical.py        # LinearReg, RandomForest, XGBoost
│   │   ├── deep_learning.py    # TensorFlow/Keras MLP
│   │   └── evaluator.py        # MAE, RMSE, R², MAPE
│   ├── api/
│   │   └── main.py             # FastAPI endpoints
│   └── utils/
│       └── config.py           # Configuration centralisée
├── app/
│   └── streamlit_app.py        # Interface web interactive
├── train.py                    # Script d'entraînement
├── Dockerfile.api
├── Dockerfile.app
├── docker-compose.yml
└── requirements.txt
```

## 🚀 Démarrage rapide (Docker)

### 1. Entraîner les modèles
```bash
docker compose --profile train up trainer
```

### 2. Lancer l'API + l'interface
```bash
docker compose up --build
```

### 3. Accéder aux services
| Service | URL |
|---|---|
| **Interface Streamlit** | http://localhost:8501 |
| **API FastAPI** | http://localhost:8000 |
| **Swagger UI** | http://localhost:8000/docs |

---

## 🐍 Démarrage local (sans Docker)

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Entraîner les modèles (~10-15 min)
python train.py

# 3. Lancer l'API (terminal 1)
uvicorn src.api.main:app --reload --port 8000

# 4. Lancer Streamlit (terminal 2)
streamlit run app/streamlit_app.py
```

---

## 📊 Modèles & Performance (estimés)

| Modèle | R² | MAE | RMSE |
|---|---|---|---|
| **Deep Learning (MLP)** | ~0.89 | ~$32k | ~$48k |
| **XGBoost** | ~0.84 | ~$38k | ~$55k |
| **Random Forest** | ~0.81 | ~$42k | ~$60k |
| **Linear Regression** | ~0.67 | ~$55k | ~$80k |

## 🧠 Architecture Deep Learning

```
Input(n_features)
→ Dense(256) + BatchNorm + ReLU + Dropout(0.3)
→ Dense(128) + BatchNorm + ReLU + Dropout(0.3)
→ Dense(64) + ReLU
→ Dense(1)  [régression]

Loss: Huber (δ=50 000) — robuste aux outliers
Optimizer: Adam + ReduceLROnPlateau
Régularisation: EarlyStopping (patience=20)
```

## 🛠️ Stack Technique

| Couche | Technologie |
|---|---|
| ML Classique | Scikit-learn, XGBoost |
| Deep Learning | TensorFlow/Keras |
| API | FastAPI + Uvicorn |
| Interface | Streamlit + Plotly |
| Containerisation | Docker + Docker Compose |

## 📡 API — Exemples

```bash
# Prédiction unique
curl -X POST "http://localhost:8000/predict?model_name=deep_learning" \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": -122.23, "latitude": 37.88,
    "housing_median_age": 41, "total_rooms": 880,
    "total_bedrooms": 129, "population": 322,
    "households": 126, "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
  }'

# Comparaison des modèles
curl http://localhost:8000/models/compare
```
