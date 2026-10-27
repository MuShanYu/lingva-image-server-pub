"""文档识别接口：stub 流水线下的响应格式和错误码（额度、远程转发在官方分支）。"""

import io
from pathlib import Path

import httpx
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app
from app.pipeline import StubPipeline

ADMIN = {"Authorization": "Bearer admin"}


def png(w: int = 300, h: int = 400, noise: bool = True) -> bytes:
    arr = np.full((h, w, 3), 255, np.uint8)
    if noise:
        arr = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def settings(tmp_path: Path, **env: str) -> Settings:
    return Settings.from_env(
        {"API_TOKEN": "admin", "USE_STUB": "1", "DATA_DIR": str(tmp_path), "DOC_PIPELINE_ENABLED": "1", **env}
    )


def post(c: TestClient, path: str, data: bytes, headers: dict[str, str] = ADMIN, **form: str) -> httpx.Response:
    return c.post(path, headers=headers, files={"file": ("p.png", data, "image/png")}, data=form)


def test_health_reports_doc_pipeline(tmp_path: Path) -> None:
    with TestClient(create_app(settings(tmp_path))) as c:
        info = c.get("/v1/health").json()["doc_pipeline"]
        assert info["enabled"] is True and info["models"]["layout"] == "stub" and info["erase"] is True
    with TestClient(create_app(settings(tmp_path, DOC_PIPELINE_ENABLED="0"))) as c:
        assert c.get("/v1/health").json()["doc_pipeline"] == {"enabled": False, "models": {}, "erase": False}
        r = post(c, "/v1/doc/page", png())
        assert r.status_code == 503 and r.json()["code"] == "doc_pipeline_disabled"


def test_doc_page_response(tmp_path: Path) -> None:
    with TestClient(create_app(settings(tmp_path))) as c:
        r = post(c, "/v1/doc/page", png(), lang_hint="en")
        assert r.status_code == 200
        body = r.json()
        assert body["mode"] == "doc_page" and body["rotation"] == 0
        assert [b["type"] for b in body["blocks"]] == ["heading", "paragraph"]
        blk = body["blocks"][1]
        assert blk["text"].startswith("Document translation") and "con- verting" not in blk["text"]
        assert all(0 <= v <= 1 for v in blk["bbox"])
        assert blk["lines"][0]["poly"] and blk["table"] is None
        assert "decode" in body["timing_ms"]
        # lang_hint 格式校验
        assert post(c, "/v1/doc/page", png(), lang_hint="en; drop").status_code == 400
        assert body["erased_image"] is None


def test_doc_page_erase(tmp_path: Path) -> None:
    with TestClient(create_app(settings(tmp_path))) as c:
        body = post(c, "/v1/doc/page", png(), erase="true").json()
        erased = Image.open(io.BytesIO(__import__("base64").b64decode(body["erased_image"])))
        assert erased.format == "WEBP" and erased.size == (300, 400)
        assert [b["type"] for b in body["blocks"]] == ["heading", "paragraph"]


def test_doc_figure_erase_and_no_text(tmp_path: Path) -> None:
    with TestClient(create_app(settings(tmp_path))) as c:
        body = post(c, "/v1/doc/figure", png(), erase="true").json()
        assert body["has_text"] is True and body["regions"][0]["text"] == "Revenue (USD)"
        erased = Image.open(io.BytesIO(__import__("base64").b64decode(body["erased_image"])))
        assert erased.format == "WEBP" and erased.size == (300, 400)
        assert post(c, "/v1/doc/figure", png()).json()["erased_image"] is None
        blank = post(c, "/v1/doc/figure", png(noise=False), erase="true").json()
        assert blank["has_text"] is False and blank["erased_image"] is None


def test_doc_errors(tmp_path: Path) -> None:
    with TestClient(create_app(settings(tmp_path, DOC_MAX_SIDE="350"))) as c:
        r = post(c, "/v1/doc/page", png(300, 400))
        assert r.status_code == 413
        assert r.json()["code"] == "image_too_large" and r.json()["max_side"] == 350
        r = post(c, "/v1/doc/page", b"not an image")
        assert r.status_code == 400 and r.json()["code"] == "invalid_image"
