"""
Modèle Deep Learning — MLP avec TensorFlow/Keras.
Architecture : Dense(256) → Dense(128) → Dense(64) → Dense(1)
Loss : Huber (robuste aux outliers de prix immobilier)
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from src.utils.config import DL_MODEL_PATH, DL_CONFIG, RANDOM_STATE
from src.models.evaluator import compute_metrics

# Reproductibilité
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def build_model(input_dim: int) -> keras.Model:
    """Construit et compile le MLP."""
    cfg = DL_CONFIG
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = inputs

    for i, units in enumerate(cfg["hidden_layers"]):
        x = layers.Dense(units, name=f"dense_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Activation("relu", name=f"relu_{i}")(x)
        x = layers.Dropout(cfg["dropout_rate"], name=f"dropout_{i}")(x)

    output = layers.Dense(1, name="output")(x)

    model = keras.Model(inputs, output, name="HousePriceMLP")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss=keras.losses.Huber(delta=50_000),   # robuste aux outliers
        metrics=["mae"],
    )
    model.summary()
    return model


def train_deep_learning(X_train, y_train, X_test, y_test) -> dict:
    """Entraîne, sauvegarde et évalue le modèle DL."""
    print("\n[DeepLearning] Construction du modèle...")
    cfg = DL_CONFIG

    model = build_model(input_dim=X_train.shape[1])

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg["patience"],
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10,
            min_lr=1e-6, verbose=1,
        ),
        callbacks.ModelCheckpoint(
            str(DL_MODEL_PATH), monitor="val_loss",
            save_best_only=True, verbose=0,
        ),
    ]

    print("[DeepLearning] Entraînement...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=cb_list,
        verbose=1,
    )

    print(f"[DeepLearning] Modèle sauvegardé → {DL_MODEL_PATH}")
    y_pred = model.predict(X_test, verbose=0).flatten()
    metrics = compute_metrics(y_pred, y_test, "Deep Learning (MLP)")
    return metrics, history.history


def load_dl_model() -> keras.Model:
    """Charge le modèle DL sauvegardé."""
    if not DL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle DL introuvable : {DL_MODEL_PATH}. Lancez train.py d'abord."
        )
    return keras.models.load_model(str(DL_MODEL_PATH))
