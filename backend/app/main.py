"""API FastAPI : transfert de style anime (photo -> Hayao / Shinkai)."""

from __future__ import annotations

import base64
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from .inference import (
    MAX_SIDE,
    MODEL_VERSION,
    STYLE_INFO,
    InvalidImageError,
    ModelNotReadyError,
    StyleTransferEngine,
    decode_image,
    png_bytes,
    side_by_side,
)
from .schemas import (
    ErrorResponse,
    HealthResponse,
    ModelMetadata,
    PredictionResponse,
    Style,
    StyleInfo,
)

engine = StyleTransferEngine()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await run_in_threadpool(engine.load)
    yield


app = FastAPI(
    title="Anime Style Transfer API",
    description="Transforme une photo en image de style anime (Hayao, Shinkai).",
    version=MODEL_VERSION,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

ERRORS = {
    400: {"model": ErrorResponse, "description": "Le fichier n'est pas une image valide"},
    503: {"model": ErrorResponse, "description": "Modèle non instancié"},
}


def load_metrics() -> dict[str, float | None]:
    """Lit models/metrics.json (écrit par src/models/evaluate.py) s'il existe."""
    metrics: dict[str, float | None] = {"fid": None, "lpips": None}
    path = Path(engine.models_dir) / "metrics.json"
    if path.is_file():
        metrics.update(json.loads(path.read_text(encoding="utf-8")))
    return metrics


async def run_inference(file: UploadFile, style: Style):
    if not engine.ready:
        raise HTTPException(status_code=503, detail="Modèle non instancié.")
    data = await file.read()
    try:
        image = decode_image(data)
        output, latency_ms = await run_in_threadpool(engine.stylize, image, style)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return image, output, latency_ms


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health() -> HealthResponse:
    """État du serveur, présence du GPU et version du modèle chargé."""
    return HealthResponse(
        status="ok" if engine.ready else "loading",
        gpu_available=engine.gpu_available(),
        model_loaded=engine.ready,
        model_version=MODEL_VERSION,
    )


@app.get(
    "/model/metadata",
    response_model=ModelMetadata,
    responses={503: ERRORS[503]},
    tags=["model"],
)
async def model_metadata() -> ModelMetadata:
    """Nom, architecture, métriques de validation et styles disponibles."""
    if not engine.ready:
        raise HTTPException(status_code=503, detail="Modèle non instancié.")
    styles = [
        StyleInfo(
            name=style,
            label=STYLE_INFO[style][0],
            description=STYLE_INFO[style][1],
            engine=engine.engine_for(style),
        )
        for style in Style
    ]
    return ModelMetadata(
        name="anime-style-transfer",
        architecture="AnimeGAN (générateur feed-forward, un modèle par style)",
        version=MODEL_VERSION,
        input_size=MAX_SIDE,
        metrics=load_metrics(),
        classes=list(Style),
        styles=styles,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses=ERRORS,
    tags=["inference"],
)
async def predict(
    file: UploadFile = File(..., description="Photo à transformer"),
    style: Style = Form(Style.hayao, description="Style d'anime cible"),
) -> PredictionResponse:
    """Reçoit une image, applique le style et retourne le résultat en JSON."""
    _, output, latency_ms = await run_inference(file, style)
    return PredictionResponse(
        style=style,
        engine=engine.engine_for(style),
        width=output.width,
        height=output.height,
        latency_ms=round(latency_ms, 2),
        image_base64=base64.b64encode(png_bytes(output)).decode("ascii"),
    )


@app.post(
    "/predict/visualize",
    responses={
        200: {"content": {"image/png": {}}, "description": "Planche avant / après"},
        **ERRORS,
    },
    response_class=Response,
    tags=["inference"],
)
async def predict_visualize(
    file: UploadFile = File(..., description="Photo à transformer"),
    style: Style = Form(Style.hayao, description="Style d'anime cible"),
) -> Response:
    """Retourne une image PNG : l'original à gauche, la version anime à droite."""
    image, output, latency_ms = await run_inference(file, style)
    return Response(
        content=png_bytes(side_by_side(image, output)),
        media_type="image/png",
        headers={"X-Inference-Latency-Ms": f"{latency_ms:.2f}"},
    )
