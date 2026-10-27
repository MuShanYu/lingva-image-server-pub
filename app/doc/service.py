"""主进程里的文档识别流程：解码、尺寸检查、交给 worker 识别、擦字。真实流水线和 stub 共用。

识别本身在文档 worker 子进程里（DocWorkerClient），这里只依赖主进程环境（Pillow、numpy）。
"""

import time
from collections.abc import Callable
from typing import Literal

import numpy as np
from PIL import Image

from ..stages.decode import decode_image
from ..stages.encode import encode_webp_b64
from .assemble import LayoutBox, OcrLine, assemble_figure, assemble_page
from .errors import ImageTooLarge

Mode = Literal["doc_page", "doc_figure"]


def load_rgb(data: bytes, max_side: int) -> np.ndarray:
    """解码（EXIF 转正）并检查长边；超过上限时抛出 ImageTooLarge，插件压缩后重试"""
    img = decode_image(data)
    if max(img.size) > max_side:
        raise ImageTooLarge(max_side)
    return np.asarray(img)


Recognizer = Callable[[str, np.ndarray, str | None], dict]  # (op, rgb, lang_hint) → worker 的结果
Eraser = Callable[[np.ndarray, list[list[list[float]]]], np.ndarray]  # (rgb, 行多边形（像素）) → 擦字后的图


def run_doc(
    mode: Mode,
    data: bytes,
    *,
    lang_hint: str | None,
    erase: bool,
    max_side: int,
    recognize: Recognizer,
    eraser: Eraser,
    webp_quality: int,
    timings: dict[str, int],
) -> dict:
    """返回 doc_page / doc_figure 的响应（dict，由接口按 schemas 校验）。timings 记录主进程里的阶段耗时"""
    t0 = time.perf_counter()
    rgb = load_rgb(data, max_side)
    timings["decode"] = round((time.perf_counter() - t0) * 1000)
    op = "page" if mode == "doc_page" else "figure"
    result = recognize(op, rgb, lang_hint)
    result["timing_ms"] = {**timings, **result.get("timing_ms", {})}
    result["erased_image"] = None
    # doc_page 擦掉所有行（含 skip、formula 块；figure 块没有行），哪些块贴回译文由插件决定。
    # 行坐标已逆变换回上传的原图，所以直接在原图上擦，转正过的页面也不用再转回去
    items = result["blocks"] if mode == "doc_page" else result["regions"]
    lines = [ln for item in items for ln in item["lines"]]
    if erase and lines:
        t0 = time.perf_counter()
        h, w = rgb.shape[:2]
        out = eraser(rgb, [[[x * w, y * h] for x, y in ln["poly"]] for ln in lines])
        result["erased_image"] = encode_webp_b64(Image.fromarray(out), webp_quality)
        result["timing_ms"]["erase"] = round((time.perf_counter() - t0) * 1000)
    return result


class FakeEngine:
    """不加载模型的假引擎（USE_STUB、DOC_WORKER_FAKE）：doc_page 返回一个标题和一段正文；
    doc_figure 在纯色图上返回无文字（用于测试不扣额度），否则返回一个区域"""

    models = {"layout": "stub", "det": "stub", "orientation": "stub", "rec": ["stub"]}

    def page(self, rgb: np.ndarray, lang_hint: str | None = None) -> dict:
        h, w = rgb.shape[:2]
        lines = [
            _line(0.08 * w, 0.06 * h, 0.6 * w, 0.09 * h, "1. Introduction"),
            _line(0.08 * w, 0.12 * h, 0.92 * w, 0.14 * h, "Document translation is not solely a problem of con-"),
            _line(0.08 * w, 0.15 * h, 0.7 * w, 0.17 * h, "verting words between languages."),
        ]
        layout = [
            LayoutBox("paragraph_title", (0.08 * w, 0.06 * h, 0.6 * w, 0.09 * h)),
            LayoutBox("text", (0.08 * w, 0.12 * h, 0.92 * w, 0.17 * h)),
        ]
        result = assemble_page(lines, layout, angle=0, upright_size=(w, h))
        return {"mode": "doc_page", "lang": "auto", "rec_model": "stub", **result, "timing_ms": {"stub": 0}}

    def figure(self, rgb: np.ndarray, lang_hint: str | None = None) -> dict:
        h, w = rgb.shape[:2]
        lines = [] if float(rgb.std()) < 1.0 else [_line(0.1 * w, 0.1 * h, 0.6 * w, 0.2 * h, "Revenue (USD)")]
        result = assemble_figure(lines, (w, h))
        return {"mode": "doc_figure", "lang": "auto", "rec_model": "stub", **result, "timing_ms": {"stub": 0}}


def _line(x1: float, y1: float, x2: float, y2: float, text: str) -> OcrLine:
    return OcrLine(((x1, y1), (x2, y1), (x2, y2), (x1, y2)), text, 0.99)
