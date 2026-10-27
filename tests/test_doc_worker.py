import base64
import io
import sys

import numpy as np
import pytest
from PIL import Image

from app.doc.client import DocWorkerClient
from app.doc.erase import text_mask
from app.doc.protocol import read_msg, write_msg
from app.doc.errors import DocInferenceError, DocTimeout, ImageTooLarge
from app.doc.service import FakeEngine, load_rgb, run_doc


@pytest.fixture
def worker():
    client = DocWorkerClient(sys.executable, timeout=20, start_timeout=60, env={"DOC_WORKER_FAKE": "1"})
    yield client
    client.close()


def _png(w: int, h: int, color=(255, 255, 255), noise: bool = False) -> bytes:
    arr = np.full((h, w, 3), color, np.uint8)
    if noise:
        arr = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_protocol_roundtrip() -> None:
    buf = io.BytesIO()
    write_msg(buf, {"id": 1, "text": "日本語"}, b"\x00\x01\x02")
    write_msg(buf, {"id": 2})
    buf.seek(0)
    assert read_msg(buf) == ({"id": 1, "text": "日本語"}, b"\x00\x01\x02")
    assert read_msg(buf) == ({"id": 2}, b"")
    assert read_msg(buf) is None


def test_worker_page_and_figure(worker: DocWorkerClient) -> None:
    assert worker.models["layout"] == "stub"
    page = worker.call("page", np.zeros((400, 300, 3), np.uint8), "en")
    assert page["mode"] == "doc_page"
    assert [b["type"] for b in page["blocks"]] == ["heading", "paragraph"]
    assert page["blocks"][1]["text"] == "Document translation is not solely a problem of converting words between languages."
    blank = worker.call("figure", np.full((100, 100, 3), 200, np.uint8), None)
    assert blank["has_text"] is False


def test_worker_timeout_restarts(worker: DocWorkerClient) -> None:
    with pytest.raises(DocTimeout):
        worker.call("sleep", None, "5", timeout=0.5)
    assert not worker.alive
    # 下一次调用自动重启
    assert worker.call("figure", np.full((50, 50, 3), 1, np.uint8), None)["has_text"] is False
    assert worker.alive


def test_worker_crash_and_error(worker: DocWorkerClient) -> None:
    with pytest.raises(DocInferenceError, match="unknown op"):
        worker.call("nope", np.zeros((2, 2, 3), np.uint8), None)
    with pytest.raises(DocInferenceError, match="退出"):
        worker.call("crash", None, None)
    assert worker.call("page", np.zeros((40, 30, 3), np.uint8), None)["mode"] == "doc_page"


def test_load_rgb_checks_long_side() -> None:
    assert load_rgb(_png(300, 200), 300).shape == (200, 300, 3)
    with pytest.raises(ImageTooLarge) as e:
        load_rgb(_png(301, 200), 300)
    assert e.value.max_side == 300


def test_text_mask_dilates_by_line_thickness() -> None:
    mask = text_mask(100, 200, [[[20, 40], [120, 40], [120, 60], [20, 60]]])
    assert mask[50, 70] == 255
    # 行粗 20，膨胀 3 像素
    assert mask[37, 70] == 255 and mask[35, 70] == 0
    assert mask[50, 123] == 255 and mask[50, 125] == 0
    assert text_mask(10, 10, []).sum() == 0


def _fake_recognize(op: str, rgb: np.ndarray, lang: str | None) -> dict:
    engine = FakeEngine()
    return engine.page(rgb, lang) if op == "page" else engine.figure(rgb, lang)


def test_run_doc_figure_erase() -> None:
    calls = []

    def eraser(rgb: np.ndarray, polys: list) -> np.ndarray:
        calls.append(polys)
        return np.zeros_like(rgb)

    kw = dict(max_side=1000, recognize=_fake_recognize, eraser=eraser, webp_quality=80)
    result = run_doc("doc_figure", _png(200, 100, noise=True), lang_hint=None, erase=True, timings={}, **kw)
    assert result["has_text"] is True
    assert result["erased_image"]
    # 多边形换算回像素坐标
    assert calls[0][0][0] == pytest.approx([20, 10])
    assert set(result["timing_ms"]) >= {"decode", "erase", "stub"}

    blank = run_doc("doc_figure", _png(200, 100), lang_hint=None, erase=True, timings={}, **kw)
    assert blank["has_text"] is False and blank["erased_image"] is None
    assert len(calls) == 1

    no_erase = run_doc("doc_figure", _png(200, 100, noise=True), lang_hint=None, erase=False, timings={}, **kw)
    assert no_erase["erased_image"] is None


def test_run_doc_page_erase_covers_all_lines() -> None:
    calls = []

    def eraser(rgb: np.ndarray, polys: list) -> np.ndarray:
        calls.append(polys)
        return np.zeros_like(rgb)

    def recognize(op, rgb, lang):
        result = FakeEngine().page(rgb, lang)
        # 页码块（skip）的行也要擦；figure 块没有行
        line = {"text": "7", "bbox": [0.5, 0.9, 0.55, 0.95], "poly": [[0.5, 0.9], [0.55, 0.9], [0.55, 0.95], [0.5, 0.95]], "conf": 1.0}
        result["blocks"] += [{"type": "skip", "lines": [line]}, {"type": "figure", "lines": []}]
        return result

    kw = dict(max_side=1000, recognize=recognize, eraser=eraser, webp_quality=80)
    result = run_doc("doc_page", _png(200, 100), lang_hint=None, erase=True, timings={}, **kw)
    assert len(calls[0]) == 4
    assert calls[0][-1][0] == pytest.approx([100, 90])
    erased = Image.open(io.BytesIO(base64.b64decode(result["erased_image"])))
    assert erased.format == "WEBP" and erased.size == (200, 100)
    assert "erase" in result["timing_ms"]

    assert run_doc("doc_page", _png(200, 100), lang_hint=None, erase=False, timings={}, **kw)["erased_image"] is None

    def no_lines(op, rgb, lang):
        return {**FakeEngine().page(rgb, lang), "blocks": [{"type": "figure", "lines": []}]}

    kw["recognize"] = no_lines
    assert run_doc("doc_page", _png(200, 100), lang_hint=None, erase=True, timings={}, **kw)["erased_image"] is None
    assert len(calls) == 1


class _SideRec:
    """假识别器：标记在裁图右侧时识别正确（置信度高），否则识别错误（置信度低）"""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, crops, batch_size):
        self.calls += 1
        for crop in crops:
            right = crop[:, -3:].mean() > crop[:, :3].mean()
            yield {"rec_text": "好" if right else "坏", "rec_score": 0.99 if right else 0.2}


def test_recognize_retries_low_score_tall_crop_upside_down() -> None:
    from app.doc.engine import DocEngine, EngineConfig

    engine = object.__new__(DocEngine)
    engine.cfg = EngineConfig()
    img = np.zeros((100, 200, 3), np.uint8)
    img[:10, 40:60] = 255  # 竖条（字形侧转的横排标题）的顶端：裁图按竖排转 90° 后在左侧，识别错误
    img[90:, 140:160] = 255  # 另一竖条的底端：转 90° 后在右侧，识别正确，不用重试
    tall = lambda x: np.array([[x, 0], [x + 20, 0], [x + 20, 100], [x, 100]], np.float32)
    rec = _SideRec()
    lines = engine._recognize(rec, img, [tall(40), tall(140)])
    assert [ln.text for ln in lines] == ["好", "好"]
    assert rec.calls == 2  # 第二次只重试置信度低的那一条
