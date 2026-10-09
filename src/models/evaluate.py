"""Évaluation (FID, LPIPS) et export ONNX d'un générateur entraîné.

Exemple :
    python -m src.models.evaluate --style Hayao --checkpoint checkpoints/hayao/generator.pt

Écrit models/<style>.onnx (lu par le backend) et met à jour models/metrics.json.
  - FID   : distance entre les images générées et les vraies images d'anime (plus bas = mieux).
  - LPIPS : distance perceptuelle entre la photo et sa version stylisée (plus bas = contenu
            mieux préservé).
Les photos utilisées ici viennent de data/raw/test_photo, jamais vues à l'entraînement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader

from src.data.dataset import PhotoDataset
from src.models.architecture import AnimeGenerator


def export_onnx(generator: torch.nn.Module, path: Path, size: int = 256) -> None:
    """Export avec hauteur et largeur dynamiques (multiples de 32 côté backend)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    generator = generator.cpu().eval()
    torch.onnx.export(
        generator,
        torch.zeros(1, 3, size, size),
        str(path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=17,
    )


def to_uint8(batch: torch.Tensor) -> torch.Tensor:
    return ((batch.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)


@torch.no_grad()
def compute_metrics(generator, photo_dir: str, anime_dir: str, size: int, device) -> dict:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    fid = FrechetInceptionDistance(feature=2048).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    generator = generator.to(device).eval()

    for anime in DataLoader(PhotoDataset(anime_dir, size), batch_size=16):
        fid.update(to_uint8(anime.to(device)), real=True)
    for photo in DataLoader(PhotoDataset(photo_dir, size), batch_size=16):
        photo = photo.to(device)
        fake = generator(photo).clamp(-1, 1)
        fid.update(to_uint8(fake), real=False)
        lpips.update(fake, photo)
    return {"fid": float(fid.compute()), "lpips": float(lpips.compute())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation et export ONNX")
    parser.add_argument("--style", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--skip-metrics", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = AnimeGenerator()
    generator.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    models_dir = Path(args.models_dir)
    metrics = {}
    if not args.skip_metrics:
        raw = Path(args.raw_dir)
        metrics = compute_metrics(
            generator, str(raw / "test_photo"), str(raw / args.style / "style"), args.size, device
        )
        metrics_path = models_dir / "metrics.json"
        models_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(metrics)

    onnx_path = models_dir / f"{args.style.lower()}.onnx"
    export_onnx(generator, onnx_path, args.size)
    print(f"Modèle exporté : {onnx_path}")

    mlflow.set_experiment("anime-style-transfer")
    with mlflow.start_run(run_name=f"evaluate-{args.style.lower()}"):
        mlflow.log_param("style", args.style)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(onnx_path), artifact_path="onnx")
        # TODO Membre A : enregistrer le modèle dans le Model Registry puis le passer en
        # Staging / Production, par ex. mlflow.onnx.log_model(..., registered_model_name=...).


if __name__ == "__main__":
    main()
