"""Wrapper d'inférence : charge un générateur ONNX par style.

Contrat avec le Membre A (export ONNX) :
  - fichier       : <MODELS_DIR>/<style>.onnx   (ex. models/hayao.onnx)
  - entrée        : float32, forme (1, 3, H, W), RGB, valeurs dans [-1, 1]
  - sortie        : float32, forme (1, 3, H, W), RGB, valeurs dans [-1, 1]
  - H et W        : dynamiques, multiples de 32

Tant qu'un fichier .onnx est absent, le style utilise un modèle factice
(filtre cartoon classique) pour que l'API et le frontend avancent sans attendre.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError

from .schemas import Style

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "0.1.0")
MAX_SIDE = int(os.getenv("MAX_SIDE", "512"))

STYLE_INFO = {
    Style.hayao: ("Hayao", "Style Miyazaki : couleurs douces et chaudes"),
    Style.shinkai: ("Shinkai", "Style Makoto Shinkai : ciels et lumières saturés"),
}


class InvalidImageError(ValueError):
    """Le fichier reçu n'est pas une image lisible."""


class ModelNotReadyError(RuntimeError):
    """Les modèles ne sont pas encore instanciés."""


def decode_image(data: bytes) -> Image.Image:
    """Décode des octets en image RGB, en respectant l'orientation EXIF."""
    if not data:
        raise InvalidImageError("Fichier vide.")
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("Le fichier n'est pas une image valide.") from exc


def resize_for_model(image: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    """Réduit l'image (plus grand côté <= max_side) avec des côtés multiples de 32."""
    w, h = image.size
    scale = min(1.0, max_side / max(w, h))
    new_w = max(32, int(round(w * scale / 32)) * 32)
    new_h = max(32, int(round(h * scale / 32)) * 32)
    return image.resize((new_w, new_h), Image.LANCZOS)


def to_tensor(image: Image.Image) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return array.transpose(2, 0, 1)[None]


def to_image(tensor: np.ndarray) -> Image.Image:
    array = (tensor[0].transpose(1, 2, 0) + 1.0) * 127.5
    # Arrondi au plus proche : astype seul tronque (254.99997 -> 254) et décale les couleurs.
    return Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8))


def dummy_stylize(image: Image.Image, style: Style) -> Image.Image:
    """Modèle factice : aplats de couleur + contours sombres, teinte selon le style."""
    smooth = image.filter(ImageFilter.MedianFilter(5))
    flat = ImageOps.posterize(smooth, 4)
    if style == Style.hayao:
        flat = ImageEnhance.Color(flat).enhance(1.25)
        gains = (1.06, 1.02, 0.92)
    else:
        flat = ImageEnhance.Color(flat).enhance(1.6)
        flat = ImageEnhance.Contrast(flat).enhance(1.15)
        gains = (0.94, 1.0, 1.12)
    edges = smooth.convert("L").filter(ImageFilter.FIND_EDGES)
    edges = edges.filter(ImageFilter.GaussianBlur(0.8))
    ink = 1.0 - np.clip(np.asarray(edges, dtype=np.float32) / 96.0, 0.0, 0.85)
    array = np.asarray(flat, dtype=np.float32) * np.array(gains, dtype=np.float32)
    array = array * ink[..., None]
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


class StyleTransferEngine:
    """Garde en mémoire une session ONNX Runtime par style."""

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = Path(models_dir)
        self.sessions: dict[Style, object] = {}
        self.ready = False

    @staticmethod
    def gpu_available() -> bool:
        try:
            import onnxruntime as ort

            return "CUDAExecutionProvider" in ort.get_available_providers()
        except ImportError:
            return False

    def load(self) -> None:
        """Charge les .onnx présents ; les styles sans fichier restent factices."""
        self.sessions = {}
        for style in Style:
            path = self.models_dir / f"{style.value}.onnx"
            if not path.is_file():
                continue
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            if self.gpu_available():
                providers.insert(0, "CUDAExecutionProvider")
            self.sessions[style] = ort.InferenceSession(str(path), providers=providers)
        self.ready = True

    def engine_for(self, style: Style) -> str:
        return "onnx" if style in self.sessions else "dummy"

    def stylize(self, image: Image.Image, style: Style) -> tuple[Image.Image, float]:
        """Retourne l'image stylisée et la latence d'inférence en millisecondes."""
        if not self.ready:
            raise ModelNotReadyError("Modèle non instancié.")
        resized = resize_for_model(image)
        start = time.perf_counter()
        session = self.sessions.get(style)
        if session is None:
            output = dummy_stylize(resized, style)
        else:
            input_name = session.get_inputs()[0].name
            result = session.run(None, {input_name: to_tensor(resized)})[0]
            output = to_image(result)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return output, latency_ms


def side_by_side(before: Image.Image, after: Image.Image) -> Image.Image:
    """Planche avant/après pour /predict/visualize."""
    before = before.resize(after.size, Image.LANCZOS)
    gap = 8
    canvas = Image.new("RGB", (after.width * 2 + gap, after.height), "white")
    canvas.paste(before, (0, 0))
    canvas.paste(after, (after.width + gap, 0))
    return canvas


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
