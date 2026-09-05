from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def test_health(client: TestClient) -> None:
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert r.headers.get("x-request-id")


def test_ready(client: TestClient) -> None:
    r = client.get("/api/v1/ready")
    assert r.status_code == 200
    assert r.json()["ready"] is True


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


def test_root_redirects_to_docs_without_spa(client: TestClient) -> None:
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (307, 308)
    assert r.headers["location"].endswith("/api/docs")
