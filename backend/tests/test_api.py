"""Tests unitaires : endpoints de l'API et fonctions d'inférence."""

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.inference import (
    InvalidImageError,
    StyleTransferEngine,
    decode_image,
    resize_for_model,
    to_image,
    to_tensor,
)
from app.schemas import Style


def make_image_bytes(size=(200, 120), fmt="JPEG") -> bytes:
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture()
def client():
    with TestClient(main.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert isinstance(body["gpu_available"], bool)
    assert body["model_version"]


def test_metadata(client):
    response = client.get("/model/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["classes"] == ["hayao", "shinkai"]
    assert {"fid", "lpips"} <= set(body["metrics"])
    assert all(s["engine"] in {"onnx", "dummy"} for s in body["styles"])


@pytest.mark.parametrize("style", ["hayao", "shinkai"])
def test_predict_returns_image(client, style):
    files = {"file": ("photo.jpg", make_image_bytes(), "image/jpeg")}
    response = client.post("/predict", files=files, data={"style": style})
    assert response.status_code == 200
    body = response.json()
    assert body["style"] == style
    assert body["latency_ms"] >= 0
    image = Image.open(io.BytesIO(base64.b64decode(body["image_base64"])))
    assert image.size == (body["width"], body["height"])
    assert image.width % 32 == 0 and image.height % 32 == 0


def test_predict_default_style(client):
    files = {"file": ("photo.png", make_image_bytes(fmt="PNG"), "image/png")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    assert response.json()["style"] == "hayao"


def test_predict_invalid_image_returns_400(client):
    files = {"file": ("notes.txt", b"ceci n'est pas une image", "text/plain")}
    response = client.post("/predict", files=files, data={"style": "hayao"})
    assert response.status_code == 400
    assert "image" in response.json()["detail"]


def test_predict_empty_file_returns_400(client):
    files = {"file": ("vide.jpg", b"", "image/jpeg")}
    assert client.post("/predict", files=files).status_code == 400


def test_predict_unknown_style_returns_422(client):
    files = {"file": ("photo.jpg", make_image_bytes(), "image/jpeg")}
    response = client.post("/predict", files=files, data={"style": "naruto"})
    assert response.status_code == 422


def test_predict_missing_file_returns_422(client):
    assert client.post("/predict", data={"style": "hayao"}).status_code == 422


def test_model_not_ready_returns_503(client, monkeypatch):
    monkeypatch.setattr(main.engine, "ready", False)
    files = {"file": ("photo.jpg", make_image_bytes(), "image/jpeg")}
    assert client.post("/predict", files=files).status_code == 503
    assert client.post("/predict/visualize", files=files).status_code == 503
    assert client.get("/model/metadata").status_code == 503
    assert client.get("/health").json()["model_loaded"] is False


def test_visualize_returns_png(client):
    files = {"file": ("photo.jpg", make_image_bytes(), "image/jpeg")}
    response = client.post("/predict/visualize", files=files, data={"style": "shinkai"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert float(response.headers["x-inference-latency-ms"]) >= 0
    image = Image.open(io.BytesIO(response.content))
    assert image.width > image.height


def test_decode_image_rejects_garbage():
    with pytest.raises(InvalidImageError):
        decode_image(b"\x00\x01\x02")


def test_resize_keeps_multiples_of_32():
    image = Image.new("RGB", (1920, 1080))
    resized = resize_for_model(image, max_side=512)
    assert max(resized.size) == 512
    assert resized.width % 32 == 0 and resized.height % 32 == 0


def test_tensor_roundtrip():
    image = Image.open(io.BytesIO(make_image_bytes(size=(64, 32), fmt="PNG")))
    tensor = to_tensor(image)
    assert tensor.shape == (1, 3, 32, 64)
    assert tensor.min() >= -1.0 and tensor.max() <= 1.0
    assert np.array_equal(np.asarray(to_image(tensor)), np.asarray(image))


def test_onnx_model_is_used_when_present(tmp_path):
    """Un fichier <style>.onnx remplace le modèle factice pour ce style."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    shape = [1, 3, "h", "w"]
    graph = helper.make_graph(
        [helper.make_node("Neg", ["input"], ["output"])],
        "negate",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, tmp_path / "hayao.onnx")

    engine = StyleTransferEngine(models_dir=tmp_path)
    engine.load()
    assert engine.engine_for(Style.hayao) == "onnx"
    assert engine.engine_for(Style.shinkai) == "dummy"

    image = Image.new("RGB", (64, 64), (255, 0, 0))
    output, latency_ms = engine.stylize(image, Style.hayao)
    assert output.getpixel((0, 0)) == (0, 255, 255)
    assert latency_ms >= 0


def test_corrupted_onnx_leaves_engine_not_ready(tmp_path):
    """Un .onnx illisible ne fait pas planter le démarrage : le moteur reste non prêt (-> 503)."""
    (tmp_path / "hayao.onnx").write_bytes(b"pas un modele onnx")
    engine = StyleTransferEngine(models_dir=tmp_path)
    engine.load()
    assert engine.ready is False
    assert engine.sessions == {}


def save_scale_model(path, factor: float) -> None:
    """Petit modèle ONNX output = input * w, avec w stocké comme poids (initializer)."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    shape = [1, 3, "h", "w"]
    weight = numpy_helper.from_array(np.array([factor], dtype=np.float32), "w")
    graph = helper.make_graph(
        [helper.make_node("Mul", ["input", "w"], ["output"])],
        "scale",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        initializer=[weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, path)


def test_mix_interpolates_generator_weights(tmp_path):
    """mix = 0.5 entre w = 0 (Hayao) et w = 2 (Shinkai) donne w = 1 : l'image ressort intacte."""
    save_scale_model(tmp_path / "hayao.onnx", 0.0)
    save_scale_model(tmp_path / "shinkai.onnx", 2.0)
    engine = StyleTransferEngine(models_dir=tmp_path)
    engine.load()
    assert engine.can_mix

    image = Image.new("RGB", (64, 64), (191, 64, 128))
    half, _ = engine.stylize(image, Style.hayao, mix=0.5)
    assert np.abs(np.asarray(half, dtype=int) - np.asarray(image, dtype=int)).max() <= 1
    hayao, _ = engine.stylize(image, Style.shinkai, mix=0.0)
    assert hayao.getpixel((0, 0)) == (128, 128, 128)  # w = 0 -> tenseur nul -> gris moyen
    assert 5 in engine.mixed_sessions  # générateur interpolé gardé en cache


def test_intensity_zero_returns_the_photo(tmp_path):
    save_scale_model(tmp_path / "hayao.onnx", 0.0)
    engine = StyleTransferEngine(models_dir=tmp_path)
    engine.load()
    image = Image.new("RGB", (64, 64), (200, 30, 90))
    output, _ = engine.stylize(image, Style.hayao, intensity=0.0)
    assert output.getpixel((10, 10)) == (200, 30, 90)


@pytest.mark.parametrize("field,value", [("mix", "1.5"), ("intensity", "-0.1"), ("intensity", "3")])
def test_slider_values_are_validated(client, field, value):
    files = {"file": ("photo.jpg", make_image_bytes(), "image/jpeg")}
    response = client.post("/predict", files=files, data={field: value})
    assert response.status_code == 422


def test_predict_accepts_sliders(client):
    files = {"file": ("photo.jpg", make_image_bytes(), "image/jpeg")}
    response = client.post("/predict", files=files, data={"mix": "0.5", "intensity": "1.5"})
    assert response.status_code == 200
    assert response.json()["intensity"] == 1.5
