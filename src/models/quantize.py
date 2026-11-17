"""Quantification INT8 statique d'un générateur ONNX (ONNX Runtime) et comparaison au FP32.

    python -m src.models.quantize --style hayao

- Calibration sur des photos d'ENTRAÎNEMENT (data/raw/train_photo) : les photos de test ne
  servent qu'à la mesure, pour ne pas biaiser la comparaison.
- Format QDQ, poids en int8 et activations en uint8.
- Mesures sur les photos de test : latence CPU moyenne, taille du fichier, écart moyen des
  sorties (en niveaux de gris sur 255) et LPIPS entre la sortie FP32 et la sortie INT8.
- Résultats ajoutés à models/metrics.json (clés quant_*_<style>). Le modèle INT8 est écrit
  dans models/<style>.int8.onnx ; le backend continue de servir le modèle FP32.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image

from src.data.dataset import list_images


def load(path: Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return array.transpose(2, 0, 1)[None]


class PhotoReader(CalibrationDataReader):
    def __init__(self, paths: list[Path], size: int) -> None:
        self.batches = iter([{"input": load(p, size)} for p in paths])

    def get_next(self):
        return next(self.batches, None)


def mean_latency_ms(session, batches: list[np.ndarray]) -> float:
    session.run(None, {"input": batches[0]})  # premier appel exclu
    start = time.perf_counter()
    for batch in batches:
        session.run(None, {"input": batch})
    return (time.perf_counter() - start) / len(batches) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantification INT8 d'un générateur ONNX")
    parser.add_argument("--style", default="hayao")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--calibration", type=int, default=64, help="photos de calibration")
    parser.add_argument("--test", type=int, default=32, help="photos de test mesurées")
    args = parser.parse_args()

    models_dir, raw = Path(args.models_dir), Path(args.raw_dir)
    fp32_path = models_dir / f"{args.style}.onnx"
    int8_path = models_dir / f"{args.style}.int8.onnx"
    prepared = models_dir / f"{args.style}.prep.onnx"

    train = list_images(raw / "train_photo")
    step = max(1, len(train) // args.calibration)
    quant_pre_process(str(fp32_path), str(prepared))
    quantize_static(
        str(prepared),
        str(int8_path),
        PhotoReader(train[::step][: args.calibration], args.size),
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
    prepared.unlink()

    providers = ["CPUExecutionProvider"]
    fp32 = ort.InferenceSession(str(fp32_path), providers=providers)
    int8 = ort.InferenceSession(str(int8_path), providers=providers)
    tests = list_images(raw / "test_photo")
    tests = tests[:: max(1, len(tests) // args.test)][: args.test]
    batches = [load(p, args.size) for p in tests]

    import torch
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex")
    diffs = []
    for batch in batches:
        a = fp32.run(None, {"input": batch})[0]
        b = int8.run(None, {"input": batch})[0]
        diffs.append(np.abs(a - b).mean() * 127.5)
        lpips.update(torch.from_numpy(np.clip(b, -1, 1)), torch.from_numpy(np.clip(a, -1, 1)))

    values = {
        "quant_latency_fp32_ms": mean_latency_ms(fp32, batches),
        "quant_latency_int8_ms": mean_latency_ms(int8, batches),
        "quant_size_fp32_mb": fp32_path.stat().st_size / 2**20,
        "quant_size_int8_mb": int8_path.stat().st_size / 2**20,
        "quant_mean_abs_diff_levels": float(np.mean(diffs)),
        "quant_lpips_fp32_vs_int8": float(lpips.compute()),
    }
    metrics_path = models_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    metrics.update({f"{key}_{args.style}": value for key, value in values.items()})
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
