# ---- Étape 1 : installation des dépendances ----
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Étape 2 : image d'exécution minimale ----
FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 MODELS_DIR=/models
WORKDIR /srv
COPY --from=builder /install /usr/local
COPY app ./app
RUN useradd --create-home appuser && mkdir -p /models && chown appuser /models
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
