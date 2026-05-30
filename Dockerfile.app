# ── Dockerfile pour l'interface Streamlit ─────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY app/ ./app/
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

# Configuration Streamlit
RUN mkdir -p ~/.streamlit
RUN echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = 8501\n\
" > ~/.streamlit/config.toml

# Exposition du port
EXPOSE 8501

# Démarrage de l'interface
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address", "0.0.0.0"]
