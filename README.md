# Anime Style Studio

Projet du module Deep Learning Avancé, **axe 2 (IA générative)** : transformer une photo
en image de style anime, avec un style au choix (Hayao, Shinkai).

- **Baseline** : CycleGAN (128×128, style Hayao)
- **Modèle avancé** : AnimeGAN (256×256, un générateur par style), exporté en ONNX
- **Métriques** : FID (proximité au style) et LPIPS (préservation du contenu)
- **Stack** : PyTorch, MLflow, DVC, FastAPI + ONNX Runtime, React + Tailwind, Docker, GitHub Actions

## Démarrage en une commande

```bash
docker compose up --build
```

| Service  | Adresse                      |
|----------|------------------------------|
| Frontend | http://localhost:3000        |
| API      | http://localhost:8000/docs   |
| MLflow   | http://localhost:5000        |

Tant que `models/hayao.onnx` et `models/shinkai.onnx` n'existent pas, l'API utilise un
**modèle factice** (filtre cartoon). L'interface indique « factice » ou « ONNX » pour chaque style.

## API

| Méthode | Route                | Rôle                                                        |
|---------|----------------------|-------------------------------------------------------------|
| GET     | `/health`            | État du serveur, présence du GPU, version du modèle         |
| GET     | `/model/metadata`    | Nom, architecture, métriques (FID, LPIPS), styles           |
| POST    | `/predict`           | `file` (image) + `style` -> JSON (image en base64, latence) |
| POST    | `/predict/visualize` | `file` + `style` -> PNG avant / après                       |

Erreurs : `400` si le fichier n'est pas une image, `503` si le modèle n'est pas instancié,
`422` si le style est inconnu.

```bash
curl -F "file=@photo.jpg" -F "style=shinkai" http://localhost:8000/predict/visualize -o resultat.png
```

## Développement local

```bash
# Backend
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements-dev.txt
MODELS_DIR=models uvicorn app.main:app --reload --app-dir backend
pytest -q && flake8 backend src && black --check backend src

# Frontend (dans un autre terminal)
cd frontend && npm install && npm run dev      # http://localhost:5173
```

Après le premier `npm install`, committez `frontend/package-lock.json`.

## Données et entraînement

```bash
pip install -r requirements-train.txt
dvc init
# Télécharger le dataset AnimeGAN dans data/raw (voir data/README.md), puis :
dvc add data/raw && git add data/raw.dvc data/.gitignore
dvc remote add -d storage gdrive://<ID_DU_DOSSIER>   # ou un dossier local / S3
dvc push

python -m src.data.dataset --style Hayao             # lissage des contours
python -m src.models.train --style Hayao --epochs 30
python -m src.models.evaluate --style Hayao --checkpoint checkpoints/hayao/generator.pt
```

`evaluate.py` écrit `models/hayao.onnx` et `models/metrics.json` : le backend les charge au
démarrage. Pour envoyer les runs vers le serveur MLflow de docker-compose :
`export MLFLOW_TRACKING_URI=http://localhost:5000`.

### Contrat d'export ONNX

Le backend attend, pour chaque style, `models/<style>.onnx` avec une entrée `float32`
`(1, 3, H, W)` RGB dans `[-1, 1]` et une sortie de même forme, H et W dynamiques.

## Arborescence

```
.github/workflows/   ci-tests.yml (lint + pytest + build front), docker-build.yml
data/                versionné par DVC
src/data/            dataset.py (dataset non apparié, lissage des contours)
src/models/          architecture.py, train.py (MLflow), evaluate.py (FID, LPIPS, ONNX)
src/utils/           visualization.py (grilles avant/après ; tient lieu de gradcam.py)
backend/             FastAPI : app/main.py, schemas.py, inference.py, tests/test_api.py
frontend/            React + Tailwind : Dropzone, Webcam, StyleSelector, BeforeAfterSlider
docker-compose.yml   backend + frontend + MLflow
dvc.yaml             pipeline prepare -> train -> evaluate
```

## Reste à faire

- [ ] Membre A : baseline CycleGAN, fine-tuning depuis des poids pré-entraînés, style Shinkai,
      enregistrement dans le MLflow Model Registry (Staging / Production)
- [ ] Membre B : remplacer les images de `frontend/public/examples` par de vraies photos
      libres de droits
- [ ] Rapport LaTeX (`rapport_technique.pdf`)

## Équipe

| Membre | Rôle                          |
|--------|-------------------------------|
| A      | Deep Learning et données      |
| B      | Full-Stack et DevOps          |
