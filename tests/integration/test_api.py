from __future__ import annotations

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from mddf.inference.predict import BBox, Prediction
from mddf.inference.registry import ArtifactsUnavailable

pytestmark = pytest.mark.integration


def _png_bytes(colour: int = 120) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (colour, colour, colour)).save(buf, format="PNG")
    return buf.getvalue()


def _fake_prediction(data: bytes, category: str, model: str, **_: object) -> Prediction:
    return Prediction(
        category=category,
        model=model,  # type: ignore[arg-type]
        model_version="test",
        verdict="defect",
        anomaly_score=12.5,
        normalized_score=0.83,
        threshold=10.0,
        latency_ms=41.2,
        heatmap_png=_png_bytes(200),
        mask_png=_png_bytes(255),
        bboxes=[BBox(x=1, y=2, width=3, height=4, score=0.9)],
        image_auroc=0.994,
    )


def test_health(client: TestClient) -> None:
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert r.headers.get("x-request-id")


def test_ready_false_before_any_export(client: TestClient) -> None:
    r = client.get("/api/v1/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is False
    assert "no exported models" in body["detail"]


def test_categories_lists_15(client: TestClient) -> None:
    r = client.get("/api/v1/categories")
    assert r.status_code == 200
    body = r.json()
    assert body["dataset"] == "mvtec_ad"
    assert len(body["categories"]) == 15
    leather = next(c for c in body["categories"] if c["name"] == "leather")
    # No models trained yet at M0.
    assert leather["available_models"] == []


def test_benchmark_empty_before_training(client: TestClient) -> None:
    r = client.get("/api/v1/benchmark")
    assert r.status_code == 200
    assert r.json()["rows"] == []


def test_metrics_endpoint(client: TestClient) -> None:
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "mddf_http_requests_total" in r.text


def test_unknown_route_returns_problem_json(client: TestClient) -> None:
    r = client.get("/api/v1/does-not-exist")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/problem+json")
    body = r.json()
    assert body["status"] == 404
    assert "type" in body and "title" in body


def test_root_serves_spa_or_redirects(client: TestClient) -> None:
    from mddf.api.main import SPA_DIR

    r = client.get("/", follow_redirects=False)
    if SPA_DIR.is_dir():
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
    else:
        assert r.status_code in (307, 308)
        assert r.headers["location"].endswith("/api/docs")


# --- predict -----------------------------------------------------------------


def test_predict_ok(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mddf.api.routes.predict.run_prediction", _fake_prediction)
    r = client.post(
        "/api/v1/predict",
        data={"category": "leather", "model": "patchcore"},
        files={"file": ("x.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] == "defect"
    assert body["category"] == "leather"
    assert body["bboxes"][0]["width"] == 3
    assert base64.b64decode(body["heatmap_png"])[:8] == b"\x89PNG\r\n\x1a\n"


def test_predict_unknown_category(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mddf.api.routes.predict.run_prediction", _fake_prediction)
    r = client.post(
        "/api/v1/predict",
        data={"category": "nope", "model": "patchcore"},
        files={"file": ("x.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/problem+json")
    assert r.json()["type"].endswith("/unknown-category")


def test_predict_rejects_non_image(client: TestClient) -> None:
    r = client.post(
        "/api/v1/predict",
        data={"category": "leather", "model": "patchcore"},
        files={"file": ("x.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 415
    assert r.json()["type"].endswith("/unsupported-media-type")


def test_predict_rejects_oversized(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MDDF_MAX_UPLOAD_BYTES", "64")
    from mddf.config import get_settings

    get_settings.cache_clear()
    try:
        r = client.post(
            "/api/v1/predict",
            data={"category": "leather", "model": "patchcore"},
            files={"file": ("x.png", _png_bytes(), "image/png")},
        )
        assert r.status_code == 413
        assert r.json()["type"].endswith("/payload-too-large")
    finally:
        get_settings.cache_clear()


def test_predict_artifacts_unavailable(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_a: object, **_k: object) -> Prediction:
        raise ArtifactsUnavailable("patchcore/leather: cannot resolve model.onnx")

    monkeypatch.setattr("mddf.api.routes.predict.run_prediction", _raise)
    r = client.post(
        "/api/v1/predict",
        data={"category": "leather", "model": "patchcore"},
        files={"file": ("x.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 503
    assert r.json()["type"].endswith("/artifacts-unavailable")


def test_predict_batch_mixed(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _sometimes(data: bytes, category: str, model: str, **_: object) -> Prediction:
        calls["n"] += 1
        if calls["n"] == 2:
            raise ArtifactsUnavailable("boom")
        return _fake_prediction(data, category, model)

    monkeypatch.setattr("mddf.api.routes.predict.run_prediction", _sometimes)
    r = client.post(
        "/api/v1/predict/batch",
        data={"category": "grid", "model": "patchcore"},
        files=[
            ("files", ("a.png", _png_bytes(), "image/png")),
            ("files", ("b.png", _png_bytes(), "image/png")),
            ("files", ("c.png", _png_bytes(), "image/png")),
        ],
    )
    assert r.status_code == 200
    items = r.json()["items"]
    assert [bool(i["result"]) for i in items] == [True, False, True]
    assert items[1]["error"]
