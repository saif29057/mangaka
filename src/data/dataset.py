"""Données non appariées photo <-> anime (dataset AnimeGAN).

Arborescence attendue après téléchargement :
    data/raw/train_photo/*.jpg          photos réelles (entraînement)
    data/raw/test_photo/*.jpg           photos réelles (évaluation, jamais vues à l'entraînement)
    data/raw/<Style>/style/*.jpg        images d'anime (ex. Hayao, Shinkai)
Prétraitement (python -m src.data.dataset --style Hayao) :
    data/processed/<Style>/smooth/*.jpg mêmes images avec contours lissés

Le lissage des contours sert à la perte adverse d'AnimeGAN : le discriminateur apprend
qu'une image d'anime aux contours flous est « fausse », ce qui force des traits nets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXTENSIONS)


def edge_smooth(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Floute uniquement les pixels proches d'un contour (Canny + dilatation)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    mask = cv2.dilate(edges, np.ones((kernel_size, kernel_size), np.uint8)) > 0
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    output = image.copy()
    output[mask] = blurred[mask]
    return output


def build_smooth_folder(style_dir: Path, out_dir: Path, size: int = 256) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = list_images(style_dir)
    for path in paths:
        image = cv2.resize(cv2.imread(str(path)), (size, size))
        cv2.imwrite(str(out_dir / path.name), edge_smooth(image))
    return len(paths)


def load_rgb(path: Path, size: int) -> np.ndarray:
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """uint8 HWC [0, 255] -> float CHW [-1, 1] (même convention que le backend)."""
    return torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)


def to_gray3(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.stack([gray, gray, gray], axis=-1)


class AnimeDataset(Dataset):
    """Retourne (photo, anime, anime en gris, anime lissé en gris), non appariés."""

    def __init__(self, raw_dir: str, processed_dir: str, style: str, size: int = 256) -> None:
        raw, processed = Path(raw_dir), Path(processed_dir)
        self.photos = list_images(raw / "train_photo")
        self.anime = list_images(raw / style / "style")
        self.smooth = list_images(processed / style / "smooth")
        self.size = size
        if not self.photos or not self.anime:
            raise FileNotFoundError(f"Images introuvables dans {raw}")
        if len(self.smooth) != len(self.anime):
            raise FileNotFoundError("Lancez d'abord : python -m src.data.dataset --style " + style)

    def __len__(self) -> int:
        return len(self.photos)

    def __getitem__(self, index: int):
        photo = load_rgb(self.photos[index], self.size)
        j = np.random.randint(len(self.anime))
        anime = load_rgb(self.anime[j], self.size)
        smooth = load_rgb(self.smooth[j], self.size)
        if np.random.rand() < 0.5:  # augmentation : miroir horizontal (train uniquement)
            photo = photo[:, ::-1].copy()
        return (
            to_tensor(photo),
            to_tensor(anime),
            to_tensor(to_gray3(anime)),
            to_tensor(to_gray3(smooth)),
        )


class PhotoDataset(Dataset):
    """Photos de test seules, pour l'évaluation (FID, LPIPS)."""

    def __init__(self, folder: str, size: int = 256) -> None:
        self.paths = list_images(Path(folder))
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        return to_tensor(load_rgb(self.paths[index], self.size))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère les images d'anime aux contours lissés")
    parser.add_argument("--style", required=True, help="Hayao ou Shinkai")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    count = build_smooth_folder(
        Path(args.raw_dir) / args.style / "style",
        Path(args.processed_dir) / args.style / "smooth",
        args.size,
    )
    print(f"{count} images lissées pour le style {args.style}")
