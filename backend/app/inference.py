"""Wrapper d'inférence : charge un générateur ONNX par style.

Contrat avec le Membre A (export ONNX) :
  - fichier       : <MODELS_DIR>/<style>.onnx   (ex. models/hayao.onnx)
  - entrée        : float32, forme (1, 3, H, W), RGB, valeurs dans [-1, 1]
  - sortie        : float32, forme (1, 3, H, W), RGB, valeurs dans [-1, 1]
  - H et W        : dynamiques, multiples de 32

Tant qu'un fichier .onnx est absent, le style utilise un modèle factice
(filtre cartoon classique) pour que l'API et le frontend avancent sans attendre.

Curseurs de l'interface (défi « interpolation » de l'axe 2) :
  - mix       : interpolation linéaire des POIDS des générateurs Hayao et Shinkai
                (0 = Hayao, 1 = Shinkai). Les deux réseaux ont la même architecture
                (Shinkai est affiné depuis Hayao), donc chaque valeur de mix définit un
                nouveau générateur, situé sur le segment qui relie les deux dans l'espace
                des paramètres. Le générateur étant feed-forward (pas d'espace latent z),
                c'est l'équivalent le plus direct d'une interpolation latente.
  - intensity : dosage du style dans l'espace image : photo + intensity * (anime - photo).
                1 = sortie du modèle ; < 1 atténue ; > 1 accentue le style.
"""

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError

from .schemas import Style

logger = logging.getLogger("uvicorn.error")

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "0.1.0")
MAX_SIDE = int(os.getenv("MAX_SIDE", "512"))
MIX_STEPS = 10  # mix arrondi au dixième : au plus 11 générateurs interpolés en cache

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


def apply_intensity(photo: Image.Image, styled: Image.Image, intensity: float) -> Image.Image:
    """photo + intensity * (styled - photo), borné à [0, 255]."""
    if intensity == 1.0:
        return styled
    base = np.asarray(photo.resize(styled.size, Image.LANCZOS), dtype=np.float32)
    anime = np.asarray(styled, dtype=np.float32)
    mixed = base + intensity * (anime - base)
    return Image.fromarray(np.clip(np.rint(mixed), 0, 255).astype(np.uint8))


def interpolate_onnx(path_a: Path, path_b: Path, t: float) -> bytes:
    """Modèle ONNX dont chaque poids flottant vaut (1 - t) * A + t * B.

    Les deux fichiers doivent avoir le même graphe (mêmes noms et formes de poids).
    """
    import onnx
    from onnx import numpy_helper

    model_a, model_b = onnx.load(str(path_a)), onnx.load(str(path_b))
    weights_b = {init.name: init for init in model_b.graph.initializer}
    for init in model_a.graph.initializer:
        other = weights_b.get(init.name)
        if other is None:
            raise ValueError(f"Poids {init.name} absent du second modèle.")
        a, b = numpy_helper.to_array(init), numpy_helper.to_array(other)
        if a.shape != b.shape:
            raise ValueError(f"Formes différentes pour {init.name}.")
        if np.issubdtype(a.dtype, np.floating):
            init.CopyFrom(
                numpy_helper.from_array(((1.0 - t) * a + t * b).astype(a.dtype), init.name)
            )
    return model_a.SerializeToString()


class StyleTransferEngine:
    """Garde en mémoire une session ONNX Runtime par style (+ les mélanges demandés)."""

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.models_dir = Path(models_dir)
        self.sessions: dict[Style, object] = {}
        self.mixed_sessions: dict[int, object] = {}
        self.ready = False

    @staticmethod
    def gpu_available() -> bool:
        try:
            import onnxruntime as ort

            return "CUDAExecutionProvider" in ort.get_available_providers()
        except ImportError:
            return False

    def load(self) -> None:
        """Charge les .onnx présents ; les styles sans fichier restent factices.

        Si un .onnx existe mais est illisible (fichier corrompu, mauvais export),
        le serveur démarre quand même mais reste « non prêt » : les routes
        d'inférence répondent 503 au lieu de faire planter l'API au démarrage.
        """
        self.sessions = {}
        self.mixed_sessions = {}
        self.ready = False
        for style in Style:
            path = self.models_dir / f"{style.value}.onnx"
            if not path.is_file():
                continue
            try:
                self.sessions[style] = self._session(str(path))
            except Exception as exc:  # onnxruntime lève plusieurs types d'erreurs
                logger.error("Impossible de charger %s : %s", path, exc)
                self.sessions = {}
                return
        self.ready = True

    def _session(self, model: str | bytes):
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if self.gpu_available():
            providers.insert(0, "CUDAExecutionProvider")
        return ort.InferenceSession(model, providers=providers)

    def engine_for(self, style: Style) -> str:
        return "onnx" if style in self.sessions else "dummy"

    @property
    def can_mix(self) -> bool:
        """Le mélange des poids exige les deux générateurs entraînés."""
        return all(style in self.sessions for style in Style)

    def _mixed_session(self, mix: float):
        """Générateur interpolé Hayao -> Shinkai, mis en cache par dixième de mix."""
        step = round(mix * MIX_STEPS)
        if step == 0:
            return self.sessions[Style.hayao]
        if step == MIX_STEPS:
            return self.sessions[Style.shinkai]
        if step not in self.mixed_sessions:
            model = interpolate_onnx(
                self.models_dir / f"{Style.hayao.value}.onnx",
                self.models_dir / f"{Style.shinkai.value}.onnx",
                step / MIX_STEPS,
            )
            self.mixed_sessions[step] = self._session(model)
        return self.mixed_sessions[step]

    def stylize(
        self,
        image: Image.Image,
        style: Style,
        mix: float | None = None,
        intensity: float = 1.0,
    ) -> tuple[Image.Image, float]:
        """Retourne l'image stylisée et la latence d'inférence en millisecondes.

        mix (0 = Hayao, 1 = Shinkai) remplace le style s'il est donné et que les deux
        modèles ONNX sont chargés ; sinon le style choisi est utilisé tel quel.
        """
        if not self.ready:
            raise ModelNotReadyError("Modèle non instancié.")
        resized = resize_for_model(image)
        start = time.perf_counter()
        if mix is not None and self.can_mix:
            session = self._mixed_session(mix)
        else:
            session = self.sessions.get(style)
        if session is None:
            output = dummy_stylize(resized, style)
        else:
            input_name = session.get_inputs()[0].name
            result = session.run(None, {input_name: to_tensor(resized)})[0]
            output = to_image(result)
        output = apply_intensity(resized, output, intensity)
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
