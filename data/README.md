# Données

Ce dossier est versionné par **DVC**, jamais par Git.

```
data/
├── raw/                    # dataset AnimeGAN tel que téléchargé  ->  dvc add data/raw
│   ├── train_photo/        # photos réelles d'entraînement
│   ├── test_photo/         # photos réelles d'évaluation (jamais vues à l'entraînement)
│   ├── Hayao/style/        # images d'anime, style Hayao
│   └── Shinkai/style/      # images d'anime, style Shinkai
└── processed/              # généré par `python -m src.data.dataset --style <Style>`
    └── <Style>/smooth/     # images d'anime aux contours lissés
```

Si l'archive téléchargée nomme autrement le dossier des photos de test, renommez-le en
`test_photo` (ou adaptez `--raw-dir` et `src/data/dataset.py`).
