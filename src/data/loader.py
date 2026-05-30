"""
Chargement et validation des données brutes.
"""
import pandas as pd
from pathlib import Path
from src.utils.config import DATA_PATH, TARGET_COLUMN


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Charge le CSV et effectue une validation minimale."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {path}")

    df = pd.read_csv(path)
    _validate(df)
    print(f"[Loader] {len(df):,} lignes chargées depuis {path.name}")
    return df


def _validate(df: pd.DataFrame) -> None:
    """Vérifie la présence des colonnes requises."""
    required = [
        "longitude", "latitude", "housing_median_age",
        "total_rooms", "total_bedrooms", "population",
        "households", "median_income", "ocean_proximity",
        TARGET_COLUMN,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
