"""
Métriques d'évaluation et rapport de comparaison des modèles.
"""
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils.config import METRICS_PATH


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray, model_name: str) -> dict:
    """Calcule MAE, RMSE, R² et MAPE."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    # MAPE (avec protection contre les zéros)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    metrics = {
        "model": model_name,
        "MAE":   round(float(mae), 2),
        "RMSE":  round(float(rmse), 2),
        "R2":    round(float(r2), 4),
        "MAPE":  round(float(mape), 2),
    }
    print(
        f"[{model_name}] R²={metrics['R2']:.4f} | "
        f"MAE=${metrics['MAE']:,.0f} | RMSE=${metrics['RMSE']:,.0f} | MAPE={metrics['MAPE']:.1f}%"
    )
    return metrics


def save_all_metrics(all_metrics: list[dict]) -> None:
    """Sauvegarde toutes les métriques dans un fichier JSON."""
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[Evaluator] Métriques sauvegardées → {METRICS_PATH}")


def load_all_metrics() -> list[dict]:
    """Charge les métriques depuis le fichier JSON."""
    if not METRICS_PATH.exists():
        return []
    with open(METRICS_PATH) as f:
        return json.load(f)


def print_leaderboard(all_metrics: list[dict]) -> None:
    """Affiche un tableau comparatif des modèles."""
    print("\n" + "=" * 65)
    print(f"{'Modèle':<22} {'R²':>8} {'MAE':>12} {'RMSE':>12} {'MAPE':>8}")
    print("-" * 65)
    sorted_metrics = sorted(all_metrics, key=lambda m: m["R2"], reverse=True)
    for m in sorted_metrics:
        print(
            f"{m['model']:<22} {m['R2']:>8.4f} "
            f"${m['MAE']:>10,.0f} ${m['RMSE']:>10,.0f} {m['MAPE']:>7.1f}%"
        )
    print("=" * 65)
