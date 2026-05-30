"""
Script principal d'entraînement.
Lance : python train.py
"""
import sys
import json
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import load_raw_data
from src.data.preprocessor import prepare_data
from src.models.classical import train_linear_regression, train_random_forest, train_xgboost
from src.models.deep_learning import train_deep_learning
from src.models.evaluator import save_all_metrics, print_leaderboard
from src.utils.config import DATA_PATH, MODELS_DIR


def main():
    print("=" * 65)
    print("   California House Price Prediction — Pipeline d'entraînement")
    print("=" * 65)

    # ── 1. Chargement ────────────────────────────────────────────────
    df = load_raw_data(DATA_PATH)

    # ── 2. Prétraitement ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test, _ = prepare_data(df)
    print(f"\nTrain : {X_train.shape} | Test : {X_test.shape}")

    # ── 3. Modèles classiques ────────────────────────────────────────
    all_metrics = []

    m1 = train_linear_regression(X_train, y_train, X_test, y_test)
    all_metrics.append(m1)

    m2 = train_random_forest(X_train, y_train, X_test, y_test)
    all_metrics.append(m2)

    m3 = train_xgboost(X_train, y_train, X_test, y_test)
    all_metrics.append(m3)

    # ── 4. Deep Learning ─────────────────────────────────────────────
    m4, dl_history = train_deep_learning(X_train, y_train, X_test, y_test)
    all_metrics.append(m4)

    # Sauvegarder l'historique DL pour Streamlit
    history_path = MODELS_DIR / "dl_history.json"
    with open(history_path, "w") as f:
        json.dump(dl_history, f)

    # ── 5. Rapport final ─────────────────────────────────────────────
    save_all_metrics(all_metrics)
    print_leaderboard(all_metrics)
    print("\n✅ Entraînement terminé ! Lancez docker-compose up pour démarrer l'app.")


if __name__ == "__main__":
    main()
