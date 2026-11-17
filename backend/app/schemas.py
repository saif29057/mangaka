"""Schémas Pydantic : tous les payloads entrants et sortants de l'API."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Style(str, Enum):
    """Styles d'anime disponibles (un générateur ONNX par style)."""

    hayao = "hayao"
    shinkai = "shinkai"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class HealthResponse(StrictModel):
    status: str = Field(examples=["ok"])
    gpu_available: bool
    model_loaded: bool
    model_version: str


class StyleInfo(StrictModel):
    name: Style
    label: str
    description: str
    engine: str = Field(description="'onnx' si un modèle entraîné est chargé, sinon 'dummy'")


class ModelMetadata(StrictModel):
    name: str
    architecture: str
    version: str
    input_size: int = Field(description="Plus grand côté de l'image envoyée au modèle")
    metrics: dict[str, Optional[float]]
    classes: list[Style] = Field(description="Les styles jouent le rôle des classes")
    styles: list[StyleInfo]
    can_mix: bool = Field(
        default=False,
        description="True si les deux générateurs ONNX sont chargés (curseur Hayao ↔ Shinkai)",
    )


class PredictionResponse(StrictModel):
    style: Style
    engine: str
    width: int
    height: int
    mix: Optional[float] = Field(
        default=None, description="Mélange des poids appliqué (0 = Hayao, 1 = Shinkai)"
    )
    intensity: float = Field(default=1.0, description="Dosage du style appliqué")
    latency_ms: float = Field(description="Temps d'inférence mesuré côté serveur")
    image_base64: str = Field(description="Image stylisée, PNG encodé en base64")


class ErrorResponse(StrictModel):
    detail: str
