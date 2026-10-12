"""Gestion du statut d'un modèle dans le MLflow Model Registry (Staging / Production).

Exemple : après validation en Staging, passer la version 1 en Production
    python -m src.models.registry --name anime-animegan-hayao --version 1 --stage Production

Deux mécanismes sont renseignés :
  - le « stage » historique (Staging / Production), celui qu'affiche l'interface MLflow
    et que demande le sujet ;
  - un alias (staging / production), qui le remplace dans les versions récentes de MLflow.
Passer une version en Production archive l'ancienne version en Production.
"""

from __future__ import annotations

import argparse
import warnings

from mlflow import MlflowClient


def set_stage(name: str, version: str | int, stage: str) -> None:
    client = MlflowClient()
    with warnings.catch_warnings():
        # Les « stages » sont marqués obsolètes par MLflow mais restent supportés.
        warnings.simplefilter("ignore", FutureWarning)
        client.transition_model_version_stage(
            name, str(version), stage, archive_existing_versions=(stage == "Production")
        )
    client.set_registered_model_alias(name, stage.lower(), str(version))
    if stage == "Production":
        # La version quitte Staging : on retire l'alias staging s'il pointait sur elle.
        aliases = client.get_model_version(name, str(version)).aliases
        if "staging" in aliases:
            client.delete_registered_model_alias(name, "staging")


def main() -> None:
    parser = argparse.ArgumentParser(description="Changer le statut d'une version de modèle")
    parser.add_argument("--name", required=True, help="ex. anime-animegan-hayao")
    parser.add_argument("--version", required=True)
    parser.add_argument("--stage", required=True, choices=["Staging", "Production"])
    args = parser.parse_args()
    set_stage(args.name, args.version, args.stage)
    print(f"{args.name} v{args.version} -> {args.stage}")


if __name__ == "__main__":
    main()
