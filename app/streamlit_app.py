"""
Interface Streamlit moderne — California House Price Prediction
Pages :
  🏠  Prédiction        — formulaire interactif
  📊  Exploration EDA   — statistiques & visualisations
  🤖  Comparaison       — benchmark tous modèles
  🧠  Deep Learning     — courbes d'entraînement
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Config de la page ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏡 California House Price AI",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles CSS custom ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f0c29, #302b63, #24243e);
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Main background */
.main { background-color: #0d1117; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e2130, #252a3d);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-label { font-size: 0.75rem; font-weight:600; color:#94a3b8; letter-spacing:0.08em; text-transform:uppercase; }
.metric-value { font-size: 2rem; font-weight:700; color:#f8fafc; margin-top:0.25rem; }
.metric-delta { font-size: 0.8rem; margin-top:0.2rem; }
.good  { color:#34d399; }
.bad   { color:#f87171; }

/* Prediction box */
.pred-box {
    background: linear-gradient(135deg, #312e81, #4338ca);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(99,102,241,0.4);
    margin: 1rem 0;
}
.pred-price {
    font-size: 3.2rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -1px;
}
.pred-label { color: #c7d2fe; font-size:0.9rem; margin-bottom:0.5rem; }

/* Section header */
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f1f5f9;
    border-left: 4px solid #6366f1;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
}

/* Badges */
.badge {
    display:inline-block; padding:0.2rem 0.7rem;
    border-radius:9999px; font-size:0.72rem; font-weight:600;
}
.badge-indigo { background:#312e81; color:#a5b4fc; }
.badge-green  { background:#064e3b; color:#6ee7b7; }

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
API_URL     = "http://api:8000"
DATA_PATH   = Path(__file__).resolve().parents[1] / "data" / "housing.csv"
MODELS_DIR  = Path(__file__).resolve().parents[1] / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"
DL_HISTORY_PATH = MODELS_DIR / "dl_history.json"

OCEAN_OPTIONS = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
MODEL_LABELS = {
    "linear":       "Linear Regression",
    "forest":       "Random Forest",
    "xgboost":      "XGBoost",
    "deep_learning":"Deep Learning (MLP)",
}
MODEL_COLORS = {
    "Linear Regression":    "#60a5fa",
    "Random Forest":        "#34d399",
    "XGBoost":              "#fb923c",
    "Deep Learning (MLP)":  "#a78bfa",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None


def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


def load_dl_history():
    if DL_HISTORY_PATH.exists():
        with open(DL_HISTORY_PATH) as f:
            return json.load(f)
    return None


def api_predict(payload: dict, model_name: str):
    try:
        r = requests.post(
            f"{API_URL}/predict",
            json=payload,
            params={"model_name": model_name},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()
        return {"error": r.json().get("detail", "Erreur API")}
    except requests.exceptions.ConnectionError:
        return {"error": "API non disponible — vérifiez que le service est démarré."}


def metric_card(label, value, delta=None, delta_good=True):
    delta_html = ""
    if delta:
        cls = "good" if delta_good else "bad"
        icon = "▲" if delta_good else "▼"
        delta_html = f'<div class="metric-delta {cls}">{icon} {delta}</div>'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)


# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏡 California House Price")
    st.markdown('<span class="badge badge-indigo">AI Dashboard v2.0</span>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Prédiction", "📊 Exploration EDA", "🤖 Comparaison Modèles", "🧠 Deep Learning"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Dataset : California Housing (Kaggle)")
    st.caption("Modèles : LinearReg · RF · XGBoost · DL")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PRÉDICTION
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Prédiction":
    st.markdown('<h1 style="color:#f1f5f9;font-size:2rem;font-weight:800;">🏠 Prédiction de Prix</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94a3b8;">Entrez les caractéristiques du logement pour obtenir une estimation instantanée.</p>', unsafe_allow_html=True)

    col_form, col_map = st.columns([1, 1], gap="large")

    with col_form:
        with st.form("prediction_form"):
            st.markdown('<div class="section-header">📍 Localisation</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            longitude = c1.number_input("Longitude", value=-118.25, min_value=-124.5, max_value=-114.0, step=0.01)
            latitude  = c2.number_input("Latitude",  value=34.05,  min_value=32.5,   max_value=42.0,   step=0.01)
            ocean_proximity = st.selectbox("Proximité océan", OCEAN_OPTIONS, index=0)

            st.markdown('<div class="section-header">🏘️ Caractéristiques</div>', unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            housing_median_age = c3.slider("Âge médian (ans)", 1, 52, 25)
            median_income      = c4.slider("Revenu médian (x$10k)", 0.5, 15.0, 4.5, step=0.1)

            c5, c6 = st.columns(2)
            total_rooms    = c5.number_input("Total pièces",  value=1500, min_value=1)
            total_bedrooms = c6.number_input("Total chambres", value=300,  min_value=1)

            c7, c8 = st.columns(2)
            population = c7.number_input("Population",  value=800,  min_value=1)
            households = c8.number_input("Ménages",     value=300,  min_value=1)

            st.markdown("---")
            model_choice = st.selectbox(
                "Modèle de prédiction",
                options=list(MODEL_LABELS.keys()),
                format_func=lambda k: MODEL_LABELS[k],
                index=3,
            )
            submitted = st.form_submit_button("🔮 Prédire le prix", use_container_width=True, type="primary")

    with col_map:
        st.markdown('<div class="section-header">🗺️ Localisation sur la carte</div>', unsafe_allow_html=True)
        df_map = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
        fig_map = px.scatter_mapbox(
            df_map, lat="lat", lon="lon",
            zoom=5, height=320,
            mapbox_style="carto-darkmatter",
        )
        fig_map.update_traces(marker=dict(size=14, color="#6366f1"))
        fig_map.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_map, use_container_width=True)

        if submitted:
            payload = {
                "longitude": longitude, "latitude": latitude,
                "housing_median_age": housing_median_age,
                "total_rooms": total_rooms, "total_bedrooms": total_bedrooms,
                "population": population, "households": households,
                "median_income": median_income, "ocean_proximity": ocean_proximity,
            }
            with st.spinner("Calcul en cours..."):
                result = api_predict(payload, model_choice)

            if "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                price = result["prediction"]
                model_used = MODEL_LABELS.get(result["model"], result["model"])
                st.markdown(f"""
                <div class="pred-box">
                    <div class="pred-label">Prix estimé par {model_used}</div>
                    <div class="pred-price">${price:,.0f}</div>
                    <div style="color:#c7d2fe;font-size:0.8rem;margin-top:0.5rem;">Valeur médiane du quartier</div>
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORATION EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Exploration EDA":
    st.markdown('<h1 style="color:#f1f5f9;font-size:2rem;font-weight:800;">📊 Exploration des Données</h1>', unsafe_allow_html=True)

    df = load_dataset()
    if df is None:
        st.error("Dataset introuvable dans /data/housing.csv")
        st.stop()

    # KPIs
    st.markdown('<div class="section-header">Vue d\'ensemble</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1: metric_card("Observations", f"{len(df):,}")
    with k2: metric_card("Prix médian", f"${df['median_house_value'].median():,.0f}")
    with k3: metric_card("Revenu médian", f"${df['median_income'].median()*10_000:,.0f}")
    with k4: metric_card("Valeurs manquantes", f"{df.isnull().sum().sum():,}", delta_good=False)

    st.markdown('<br>', unsafe_allow_html=True)

    # Distribution des prix
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Distribution des Prix</div>', unsafe_allow_html=True)
        fig = px.histogram(
            df, x="median_house_value", nbins=60,
            color_discrete_sequence=["#6366f1"],
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, xaxis_title="Prix médian ($)", yaxis_title="Fréquence",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Prix par Proximité Océan</div>', unsafe_allow_html=True)
        fig2 = px.box(
            df, x="ocean_proximity", y="median_house_value",
            color="ocean_proximity",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            template="plotly_dark",
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, xaxis_title="", yaxis_title="Prix médian ($)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Carte géographique
    st.markdown('<div class="section-header">🗺️ Carte des Prix — Californie</div>', unsafe_allow_html=True)
    df_sample = df.sample(min(5000, len(df)), random_state=42)
    fig_geo = px.scatter_mapbox(
        df_sample,
        lat="latitude", lon="longitude",
        color="median_house_value",
        size="population",
        color_continuous_scale="Plasma",
        size_max=12, zoom=5, height=500,
        mapbox_style="carto-darkmatter",
        hover_data=["median_house_value", "median_income", "ocean_proximity"],
        labels={"median_house_value": "Prix ($)"},
    )
    fig_geo.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_geo, use_container_width=True)

    # Heatmap corrélation
    st.markdown('<div class="section-header">Matrice de Corrélation</div>', unsafe_allow_html=True)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[num_cols].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        template="plotly_dark", height=500,
    )
    fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_corr, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — COMPARAISON MODÈLES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Comparaison Modèles":
    st.markdown('<h1 style="color:#f1f5f9;font-size:2rem;font-weight:800;">🤖 Benchmark des Modèles</h1>', unsafe_allow_html=True)

    metrics = load_metrics()
    if not metrics:
        st.warning("⏳ Aucune métrique disponible. Lancez d'abord `python train.py`.")
        st.stop()

    df_m = pd.DataFrame(metrics).sort_values("R2", ascending=False)
    df_m["color"] = df_m["model"].map(MODEL_COLORS).fillna("#94a3b8")

    # Podium
    best = df_m.iloc[0]
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1e1b4b,#312e81);border-radius:16px;
                padding:1.5rem 2rem;margin-bottom:1.5rem;border:1px solid rgba(99,102,241,0.3);">
        <div style="color:#a5b4fc;font-size:0.8rem;font-weight:600;text-transform:uppercase;">🏆 Meilleur Modèle</div>
        <div style="color:#fff;font-size:1.8rem;font-weight:800;margin-top:0.25rem;">{best['model']}</div>
        <div style="color:#c7d2fe;margin-top:0.25rem;">R² = {best['R2']:.4f} · MAE = ${best['MAE']:,.0f} · RMSE = ${best['RMSE']:,.0f}</div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">Score R²</div>', unsafe_allow_html=True)
        fig_r2 = px.bar(
            df_m, x="model", y="R2",
            color="model", color_discrete_map=MODEL_COLORS,
            text=df_m["R2"].apply(lambda v: f"{v:.4f}"),
            template="plotly_dark",
        )
        fig_r2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, xaxis_title="", yaxis_title="R²",
            yaxis_range=[0, 1.05],
        )
        fig_r2.update_traces(textposition="outside")
        st.plotly_chart(fig_r2, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">MAE (Erreur Absolue Moyenne)</div>', unsafe_allow_html=True)
        fig_mae = px.bar(
            df_m, x="model", y="MAE",
            color="model", color_discrete_map=MODEL_COLORS,
            text=df_m["MAE"].apply(lambda v: f"${v:,.0f}"),
            template="plotly_dark",
        )
        fig_mae.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, xaxis_title="", yaxis_title="MAE ($)",
        )
        fig_mae.update_traces(textposition="outside")
        st.plotly_chart(fig_mae, use_container_width=True)

    # Radar chart
    st.markdown('<div class="section-header">Comparaison Multi-Métriques (Radar)</div>', unsafe_allow_html=True)

    # Normaliser pour le radar (R² → plus grand = mieux, MAE/RMSE → plus petit = mieux)
    df_radar = df_m.copy()
    df_radar["MAE_inv"]  = 1 - (df_radar["MAE"]  / df_radar["MAE"].max())
    df_radar["RMSE_inv"] = 1 - (df_radar["RMSE"] / df_radar["RMSE"].max())

    categories = ["R²", "MAE (inv)", "RMSE (inv)", "MAPE (inv)"]
    df_radar["MAPE_inv"] = 1 - (df_radar["MAPE"] / df_radar["MAPE"].max())

    fig_radar = go.Figure()
    for _, row in df_radar.iterrows():
        vals = [row["R2"], row["MAE_inv"], row["RMSE_inv"], row["MAPE_inv"]]
        vals += [vals[0]]  # fermer le polygone
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=categories + [categories[0]],
            fill="toself", name=row["model"],
            line=dict(color=MODEL_COLORS.get(row["model"], "#94a3b8")),
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#334155"),
            angularaxis=dict(gridcolor="#334155"),
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#e2e8f0")),
        height=400,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Tableau récapitulatif
    st.markdown('<div class="section-header">Tableau Récapitulatif</div>', unsafe_allow_html=True)
    df_display = df_m[["model", "R2", "MAE", "RMSE", "MAPE"]].copy()
    df_display.columns = ["Modèle", "R²", "MAE ($)", "RMSE ($)", "MAPE (%)"]
    st.dataframe(
        df_display.style
            .format({"R²": "{:.4f}", "MAE ($)": "${:,.0f}", "RMSE ($)": "${:,.0f}", "MAPE (%)": "{:.1f}%"})
            .background_gradient(subset=["R²"], cmap="Greens")
            .background_gradient(subset=["MAE ($)", "RMSE ($)"], cmap="Reds_r"),
        use_container_width=True,
        hide_index=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DEEP LEARNING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Deep Learning":
    st.markdown('<h1 style="color:#f1f5f9;font-size:2rem;font-weight:800;">🧠 Deep Learning — MLP</h1>', unsafe_allow_html=True)

    # Architecture info
    st.markdown('<div class="section-header">Architecture du Réseau</div>', unsafe_allow_html=True)
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a: metric_card("Couches cachées", "3 (256-128-64)")
    with col_b: metric_card("Activation", "ReLU + BatchNorm")
    with col_c: metric_card("Dropout", "30%")
    with col_d: metric_card("Loss Function", "Huber (δ=50k)")

    st.markdown('<br>', unsafe_allow_html=True)
    col_e, col_f, col_g = st.columns(3)
    with col_e: metric_card("Optimiseur", "Adam (lr=1e-3)")
    with col_f: metric_card("Early Stopping", "Patience = 20")
    with col_g: metric_card("Batch Size", "256")

    # Courbes d'entraînement
    history = load_dl_history()
    if history:
        st.markdown('<div class="section-header">📉 Courbes d\'Entraînement</div>', unsafe_allow_html=True)

        epochs = list(range(1, len(history["loss"]) + 1))
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=epochs, y=history["loss"],
            name="Train Loss", line=dict(color="#6366f1", width=2),
        ))
        fig_loss.add_trace(go.Scatter(
            x=epochs, y=history["val_loss"],
            name="Val Loss", line=dict(color="#f472b6", width=2, dash="dash"),
        ))
        fig_loss.update_layout(
            title="Loss (Huber) par Epoch",
            xaxis_title="Epoch", yaxis_title="Loss",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(color="#e2e8f0")),
        )
        st.plotly_chart(fig_loss, use_container_width=True)

        # MAE history
        if "mae" in history:
            fig_mae = go.Figure()
            fig_mae.add_trace(go.Scatter(
                x=epochs, y=history["mae"],
                name="Train MAE", line=dict(color="#34d399", width=2),
            ))
            if "val_mae" in history:
                fig_mae.add_trace(go.Scatter(
                    x=epochs, y=history["val_mae"],
                    name="Val MAE", line=dict(color="#fb923c", width=2, dash="dash"),
                ))
            fig_mae.update_layout(
                title="MAE par Epoch",
                xaxis_title="Epoch", yaxis_title="MAE ($)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(fig_mae, use_container_width=True)

        # Résumé entraînement
        final_train = history["loss"][-1]
        best_val    = min(history["val_loss"])
        n_epochs    = len(history["loss"])
        st.info(
            f"✅ **Entraînement terminé** en **{n_epochs} epochs** · "
            f"Train Loss finale : **{final_train:,.1f}** · "
            f"Best Val Loss : **{best_val:,.1f}**"
        )
    else:
        st.warning("⏳ L'historique d'entraînement n'est pas encore disponible. Lancez `python train.py` d'abord.")

    # Explication architecture
    with st.expander("📖 Pourquoi cette architecture ?"):
        st.markdown("""
        | Choix | Raison |
        |---|---|
        | **Huber Loss** | Plus robuste que MSE aux outliers de prix (500k$+) |
        | **BatchNormalization** | Accélère la convergence, stabilise l'entraînement |
        | **Dropout 30%** | Réduit le surapprentissage sur ce dataset de taille modeste |
        | **ReduceLROnPlateau** | Adapte le learning rate si le val_loss stagne |
        | **EarlyStopping** | Évite l'overfitting et réduit le temps d'entraînement |
        | **3 couches 256→128→64** | Architecture pyramidale standard pour la régression tabulaire |
        """)
