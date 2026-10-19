import base64
import io
import re
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
HEX = re.compile(r"^#[0-9a-f]{6}$")


def make_client(**env: str) -> TestClient:
    settings = Settings.from_env({"API_TOKEN": TOKEN, "USE_STUB": "1", **env})
    return TestClient(create_app(settings))


@pytest.fixture
def client() -> Iterator[TestClient]:
    with make_client() as c:
        yield c


def image_bytes(w: int = 400, h: int = 600, fmt: str = "PNG", orientation: int | None = None) -> bytes:
    img = Image.new("RGB", (w, h), (200, 180, 160))
    buf = io.BytesIO()
    if orientation is None:
        img.save(buf, format=fmt)
    else:
        exif = Image.Exif()
        exif[0x0112] = orientation
        img.save(buf, format=fmt, exif=exif)
    return buf.getvalue()


def analyze(client: TestClient, data: bytes, headers: dict[str, str] | None = None, **form: str):
    return client.post(
        "/v1/analyze",
        headers=AUTH if headers is None else headers,
        files={"file": ("page.png", data, "application/octet-stream")},
        data=form,
    )


def decode_b64_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def test_health_needs_no_auth(client: TestClient) -> None:
    r = client.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["api_version"] == 1
    assert body["stub"] is True
    assert body["device"] == "cpu"
    assert set(body["model_version"]) == {"text_detector", "bubble_detector", "ocr", "inpainter"}
    assert body["capabilities"] == {"inpaint": True, "upscale": False}


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN}])
def test_analyze_rejects_bad_token(client: TestClient, headers: dict[str, str]) -> None:
    r = analyze(client, image_bytes(), headers=headers)
    assert r.status_code == 401
    assert r.json()["code"] == "unauthorized" and r.json()["error"]


def test_analyze_contract(client: TestClient) -> None:
    r = analyze(client, image_bytes(400, 600))
    assert r.status_code == 200
    body = r.json()
    assert (body["width"], body["height"]) == (400, 600)
    assert body["model_version"]["ocr"] == "stub"
    assert body["debug_mask"] is None
    assert {"decode", "encode"} <= set(body["timings_ms"])

    blocks = body["blocks"]
    assert [b["order"] for b in blocks] == list(range(len(blocks)))
    kinds = {b["kind"]: b for b in blocks}
    assert set(kinds) == {"bubble", "free"}
    assert kinds["bubble"]["vertical"] is True
    assert kinds["bubble"]["bubble_bbox"] is not None
    assert kinds["free"]["bubble_bbox"] is None

    for b in blocks:
        x1, y1, x2, y2 = b["bbox"]
        assert 0 <= x1 < x2 <= 400 and 0 <= y1 < y2 <= 600
        assert b["text"]
        assert b["lines"] and all(len(line["bbox"]) == 4 for line in b["lines"])
        assert b["font_size"] > 0
        assert HEX.match(b["fg_color"]) and HEX.match(b["bg_color"])

    clean = decode_b64_image(body["clean_image"])
    assert clean.format == "WEBP"
    assert clean.size == (400, 600)


def test_analyze_without_inpaint(client: TestClient) -> None:
    r = analyze(client, image_bytes(), inpaint="false")
    assert r.status_code == 200
    assert r.json()["clean_image"] is None


def test_analyze_debug_mask(client: TestClient) -> None:
    r = analyze(client, image_bytes(300, 500), debug="true")
    assert r.status_code == 200
    mask = decode_b64_image(r.json()["debug_mask"])
    assert mask.format == "PNG"
    assert mask.size == (300, 500)


def test_analyze_applies_exif_orientation(client: TestClient) -> None:
    # orientation=6 表示需顺时针旋转 90°，转正后宽高互换
    r = analyze(client, image_bytes(400, 600, fmt="JPEG", orientation=6))
    assert r.status_code == 200
    assert (r.json()["width"], r.json()["height"]) == (600, 400)


def test_analyze_accepts_webp(client: TestClient) -> None:
    assert analyze(client, image_bytes(fmt="WEBP")).status_code == 200


@pytest.mark.parametrize("data", [b"not an image", image_bytes(fmt="GIF")])
def test_analyze_rejects_undecodable(client: TestClient, data: bytes) -> None:
    r = analyze(client, data)
    assert r.status_code == 400
    assert r.json()["code"] == "invalid_image"


def test_analyze_missing_file(client: TestClient) -> None:
    r = client.post("/v1/analyze", headers=AUTH, data={"inpaint": "true"})
    assert r.status_code == 400
    assert r.json()["code"] == "invalid_request" and r.json()["fields"] == ["body.file"]


def test_analyze_rejects_too_large() -> None:
    with make_client(MAX_UPLOAD_MB="1") as c:
        r = analyze(c, b"\0" * (1024 * 1024 + 1))
    assert r.status_code == 413
    assert r.json()["code"] == "image_too_large" and r.json()["max_mb"] == 1


def test_real_pipeline_requires_token() -> None:
    with pytest.raises(ValueError):
        Settings.from_env({"USE_STUB": "0"})


def test_invalid_device_rejected() -> None:
    with pytest.raises(ValueError):
        Settings.from_env({"DEVICE": "tpu"})
