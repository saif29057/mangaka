"""Évaluation (FID, LPIPS), export ONNX et enregistrement MLflow d'un générateur entraîné.

Exemples :
    # Modèle avancé (AnimeGAN 256x256) : exporté vers models/hayao.onnx, lu par le backend
    python -m src.models.evaluate --style Hayao --checkpoint checkpoints/hayao/generator.pt
    # Baseline (CycleGAN 128x128) : exportée vers models/cyclegan-hayao.onnx (comparaison)
    python -m src.models.evaluate --arch cyclegan --size 128 --style Hayao \\
        --checkpoint checkpoints/cyclegan-hayao/generator.pt

Métriques écrites dans models/metrics.json (clés plates, lues par GET /model/metadata) :
  - FID   : distance entre les images générées et les vraies images d'anime (plus bas = mieux).
  - LPIPS : distance perceptuelle entre la photo et sa version stylisée (plus bas = contenu
            mieux préservé).
  - onnx_max_abs_diff : écart maximal entre les sorties PyTorch et ONNX Runtime (même image).
  - latency_*_ms      : latence moyenne d'inférence sur CPU, PyTorch contre ONNX Runtime.
Les photos utilisées ici viennent de data/raw/test_photo, jamais vues à l'entraînement.

Avec --register, le modèle ONNX est enregistré dans le MLflow Model Registry
(nom : anime-<arch>-<style>) avec le statut donné par --stage (Staging par défaut).
Le passage en Production se fait ensuite avec src/models/registry.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import PhotoDataset
from src.models.architecture import AnimeGenerator, ResnetGenerator

ARCHITECTURES = {"animegan": AnimeGenerator, "cyclegan": ResnetGenerator}


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
        dynamo=False,  # exporteur classique (TorchScript) : gère dynamic_axes simplement
    )


def to_uint8(batch: torch.Tensor) -> torch.Tensor:
    return ((batch.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)


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


@torch.no_grad()
def check_onnx(generator, onnx_path: Path, image: torch.Tensor, runs: int = 10) -> dict:
    """Compare PyTorch et ONNX Runtime sur la même image, et mesure la latence CPU."""
    import onnxruntime as ort

    generator = generator.cpu().eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    batch = image[None].cpu()
    torch_out = generator(batch).numpy()
    onnx_out = session.run(None, {"input": batch.numpy()})[0]

    def mean_ms(fn) -> float:
        fn()  # premier appel exclu (allocations, mise en cache)
        start = time.perf_counter()
        for _ in range(runs):
            fn()
        return (time.perf_counter() - start) / runs * 1000.0

    return {
        "onnx_max_abs_diff": float(np.abs(torch_out - onnx_out).max()),
        "latency_torch_cpu_ms": mean_ms(lambda: generator(batch)),
        "latency_onnx_cpu_ms": mean_ms(lambda: session.run(None, {"input": batch.numpy()})),
    }


def update_metrics_file(path: Path, prefix: str, values: dict, is_deployed: bool) -> dict:
    """Ajoute les métriques de ce modèle à metrics.json sans effacer celles des autres.

    Les clés restent plates (ex. fid_cyclegan_hayao) car l'API attend dict[str, float].
    Les clés « fid » et « lpips » décrivent le modèle servi par le backend.
    """
    metrics = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    metrics.update({f"{key}_{prefix}": value for key, value in values.items()})
    if is_deployed:
        metrics.update({k: values[k] for k in ("fid", "lpips") if k in values})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation et export ONNX")
    parser.add_argument("--style", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES), default="animegan")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--skip-metrics", action="store_true", help="Export ONNX seulement")
    parser.add_argument("--register", action="store_true", help="Model Registry MLflow")
    parser.add_argument("--stage", default="Staging", choices=["Staging", "Production"])
    parser.add_argument("--experiment", default="anime-style-transfer")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = ARCHITECTURES[args.arch]()
    generator.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    style = args.style.lower()
    prefix = f"{args.arch}_{style}"
    raw = Path(args.raw_dir)
    models_dir = Path(args.models_dir)

    values = {}
    if not args.skip_metrics:
        values.update(
            compute_metrics(
                generator,
                str(raw / "test_photo"),
                str(raw / args.style / "style"),
                args.size,
                device,
            )
        )

    # Le modèle avancé remplace le modèle factice du backend ; la baseline a un autre nom.
    is_deployed = args.arch == "animegan"
    onnx_name = f"{style}.onnx" if is_deployed else f"cyclegan-{style}.onnx"
    onnx_path = models_dir / onnx_name
    export_onnx(generator, onnx_path, args.size)
    print(f"Modèle exporté : {onnx_path}")

    test_image = PhotoDataset(str(raw / "test_photo"), args.size)[0]
    values.update(check_onnx(generator, onnx_path, test_image))
    update_metrics_file(models_dir / "metrics.json", prefix, values, is_deployed)
    print(json.dumps(values, indent=2))

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"evaluate-{args.arch}-{style}"):
        mlflow.log_params({"style": args.style, "arch": args.arch, "size": args.size})
        mlflow.log_metrics(values)
        if args.register:
            import onnx

            from src.models.registry import set_stage

            name = f"anime-{args.arch}-{style}"
            info = mlflow.onnx.log_model(
                onnx.load(str(onnx_path)), artifact_path="onnx", registered_model_name=name
            )
            version = info.registered_model_version
            set_stage(name, version, args.stage)
            print(f"Model Registry : {name} v{version} -> {args.stage}")
        else:
            mlflow.log_artifact(str(onnx_path), artifact_path="onnx")


if __name__ == "__main__":
    main()
