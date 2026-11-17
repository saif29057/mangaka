# Mangaka

> Mangaka : transfert de style photo → anime (Hayao, Shinkai) avec CycleGAN et AnimeGAN, PyTorch, ONNX, FastAPI, React, DVC, MLflow, Docker.

Projet du module Deep Learning Avancé, **axe 2 (IA générative)** : transformer une photo
en image de style anime, avec un style au choix (Hayao, Shinkai).

- **Baseline** : CycleGAN (128×128, style Hayao)
- **Modèle avancé** : AnimeGAN (256×256, un générateur par style), exporté en ONNX
- **Métriques** : FID (proximité au style), LPIPS (préservation du contenu), Inception Score
- **Curseurs d'interpolation** : mélange des poids Hayao ↔ Shinkai et intensité du style
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

Les modèles entraînés sont versionnés par DVC : récupérez-les avec `dvc pull` (ou `dvc checkout`
sur la machine qui les a produits) avant `docker compose up`. Sans `models/hayao.onnx` et
`models/shinkai.onnx`, l'API utilise un **modèle factice** (filtre cartoon) ; l'interface
indique « factice » ou « ONNX » pour chaque style.

## API

| Méthode | Route                | Rôle                                                        |
|---------|----------------------|-------------------------------------------------------------|
| GET     | `/health`            | État du serveur, présence du GPU, version du modèle         |
| GET     | `/model/metadata`    | Nom, architecture, métriques (FID, LPIPS), styles           |
| POST    | `/predict`           | `file` + `style` (+ `mix`, `intensity`) -> JSON (image base64, latence) |
| POST    | `/predict/visualize` | mêmes champs -> PNG avant / après                           |

Curseurs : `mix` (0 = Hayao, 1 = Shinkai) interpole les **poids** des deux générateurs ONNX
(même architecture, Shinkai est affiné depuis Hayao) ; `intensity` (0 à 2) dose le style :
`photo + intensity × (anime − photo)`.

Erreurs : `400` si le fichier n'est pas une image, `503` si le modèle n'est pas instancié,
`422` si le style est inconnu ou si un curseur sort de ses bornes.

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
python -m src.models.quantize --style hayao           # INT8 + comparaison au FP32
python -m src.models.registry --name anime-animegan-hayao --version 1 --stage Production
```

En pratique, l'entraînement a été fait sur GPU Kaggle avec `notebooks/kaggle_train.ipynb`
(deux runs, la limite d'une session étant de 12 h ; le notebook sait reprendre un run coupé).
Les poids, les ONNX et la base MLflow sont dans le `resultats.zip` produit par le notebook.

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
src/models/          architecture.py, train.py, train_cyclegan.py, evaluate.py (FID, LPIPS, IS,
                     ONNX), quantize.py (INT8), registry.py (Staging / Production)
notebooks/           kaggle_train.ipynb (entraînement GPU, reprise après coupure)
rapport/             rapport_technique.tex + figures/
src/utils/           visualization.py (grilles avant/après ; tient lieu de gradcam.py)
backend/             FastAPI : app/main.py, schemas.py, inference.py, tests/test_api.py
frontend/            React + Tailwind : Dropzone, Webcam, StyleSelector, StyleSliders,
                     BeforeAfterSlider, LatencyBadge
docker-compose.yml   backend + frontend + MLflow
dvc.yaml             pipeline prepare -> train -> evaluate
```

## Résultats (photos de test, jamais vues à l'entraînement)

| Modèle             | Résolution | FID ↓ | LPIPS ↓ |
|--------------------|-----------:|------:|--------:|
| CycleGAN (Hayao)   | 128        | 144,1 | 0,128   |
| AnimeGAN (Hayao)   | 256        | 180,1 | 0,041   |
| AnimeGAN (Shinkai) | 256        | 136,8 | 0,029   |

Détails, Inception Score, latences et analyse : `rapport/rapport_technique.tex`.

## Crédits des photos d'exemple

Les 4 photos de `frontend/public/examples` viennent de Wikimedia Commons, sous licence
**CC0** (domaine public), redimensionnées à 800 px de large.

| Fichier         | Sujet                       | Auteur           | Source |
|-----------------|-----------------------------|------------------|--------|
| `exemple-1.jpg` | Lac et montagnes            | Bonnie Moreland  | [Lake Mountain Landscape.jpg](https://commons.wikimedia.org/wiki/File:Lake_Mountain_Landscape.jpg) |
| `exemple-2.jpg` | Rue d'Amsterdam la nuit     | Fons Heijnsbroek | [Koningsstraat / Nieuwmarkt, Amsterdam](https://commons.wikimedia.org/wiki/File:2024_-_Photo_of_the_street_Koningsstraat_and_square_Nieuwmarkt_with_Christmas_lights_by_night_in_Amsterdam_city_in_winter_-_free_image_of_street_photography_in_The_Netherlands_by_Fons_Heijnsbroek.jpg) |
| `exemple-3.jpg` | Portrait de chat            | Anton Petrov     | [Cat portrait (Unsplash).jpg](https://commons.wikimedia.org/wiki/File:Cat_portrait_(Unsplash).jpg) |
| `exemple-4.jpg` | Maison traditionnelle       | Ted Moravec      | [A traditional house - Flickr - Ted Moravec.jpg](https://commons.wikimedia.org/wiki/File:A_traditional_house_-_Flickr_-_Ted_Moravec.jpg) |

## Équipe

| Membre | Rôle                          |
|--------|-------------------------------|
| A      | Deep Learning et données      |
| B      | Full-Stack et DevOps          |
