"""Visualisations (équivalent de utils/gradcam.py pour un projet génératif) :
grilles photo / résultat journalisées dans MLflow à chaque époque."""

import numpy as np
import torch
from PIL import Image


def tensor_to_uint8(batch: torch.Tensor) -> np.ndarray:
    """(N, 3, H, W) dans [-1, 1] -> (N, H, W, 3) uint8."""
    array = (batch.detach().cpu().clamp(-1, 1) + 1.0) * 127.5
    return array.permute(0, 2, 3, 1).numpy().astype(np.uint8)


def comparison_grid(photos: torch.Tensor, fakes: torch.Tensor, max_items: int = 8) -> Image.Image:
    """Ligne du haut : photos d'entrée. Ligne du bas : sorties du générateur."""
    top = np.concatenate(list(tensor_to_uint8(photos[:max_items])), axis=1)
    bottom = np.concatenate(list(tensor_to_uint8(fakes[:max_items])), axis=1)
    return Image.fromarray(np.concatenate([top, bottom], axis=0))
