import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol

from PIL import Image, ImageDraw

from .config import Settings
from .schemas import AnalyzeResponse, BBox, Block, Capabilities, Line
from .stages.decode import decode_image
from .stages.encode import encode_png_b64, encode_webp_b64


class Pipeline(Protocol):
    stub: bool
    device: str
    model_version: dict[str, str]
    capabilities: Capabilities

    def run(self, data: bytes, *, inpaint: bool, debug: bool) -> AnalyzeResponse: ...


class StageTimer:
    def __init__(self) -> None:
        self.timings: dict[str, int] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = round((time.perf_counter() - start) * 1000)


def _frac_box(w: int, h: int, fx1: float, fy1: float, fx2: float, fy2: float) -> BBox:
    return (round(w * fx1), round(h * fy1), round(w * fx2), round(h * fy2))


def _stub_blocks(w: int, h: int) -> list[Block]:
    bx1, by1, bx2, by2 = _frac_box(w, h, 0.72, 0.08, 0.88, 0.30)
    mid = (bx1 + bx2) // 2
    bubble = Block(
        id=0,
        order=0,
        kind="bubble",
        bbox=(bx1, by1, bx2, by2),
        bubble_bbox=_frac_box(w, h, 0.66, 0.04, 0.94, 0.34),
        vertical=True,
        text="なんだって…！？",
        # 竖排从右往左读，右边那列在前
        lines=[Line(bbox=(mid, by1, bx2, by2)), Line(bbox=(bx1, by1, mid, by2))],
        font_size=max(1, mid - bx1),
        fg_color="#000000",
        bg_color="#ffffff",
    )
    fx1, fy1, fx2, fy2 = _frac_box(w, h, 0.08, 0.86, 0.52, 0.92)
    free = Block(
        id=1,
        order=1,
        kind="free",
        bbox=(fx1, fy1, fx2, fy2),
        bubble_bbox=None,
        vertical=False,
        text="そのころ、町では――",
        lines=[Line(bbox=(fx1, fy1, fx2, fy2))],
        font_size=max(1, fy2 - fy1),
        fg_color="#000000",
        bg_color="#ffffff",
    )
    return [bubble, free]


def _fill_boxes(img: Image.Image, boxes: list[BBox], fill: str | int) -> None:
    draw = ImageDraw.Draw(img)
    for x1, y1, x2, y2 in boxes:
        if x2 > x1 and y2 > y1:
            draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=fill)


class StubPipeline:
    """不加载模型，按图片尺寸比例返回 2 个假块；底图为把块区域涂白的原图。"""

    stub = True
    device = "cpu"
    model_version = {
        "text_detector": "stub",
        "bubble_detector": "stub",
        "ocr": "stub",
        "inpainter": "stub",
    }
    capabilities = Capabilities(inpaint=True, upscale=False)

    def __init__(self, settings: Settings) -> None:
        self._webp_quality = settings.webp_quality

    def run(self, data: bytes, *, inpaint: bool, debug: bool) -> AnalyzeResponse:
        timer = StageTimer()
        with timer.stage("decode"):
            img = decode_image(data)
        w, h = img.size
        with timer.stage("stub"):
            blocks = _stub_blocks(w, h)
            boxes = [b.bbox for b in blocks]

        clean_image: str | None = None
        debug_mask: str | None = None
        with timer.stage("encode"):
            if inpaint:
                _fill_boxes(img, boxes, "white")
                clean_image = encode_webp_b64(img, self._webp_quality)
            if debug:
                mask = Image.new("L", (w, h), 0)
                _fill_boxes(mask, boxes, 255)
                debug_mask = encode_png_b64(mask)

        return AnalyzeResponse(
            width=w,
            height=h,
            model_version=dict(self.model_version),
            blocks=blocks,
            clean_image=clean_image,
            debug_mask=debug_mask,
            timings_ms=timer.timings,
        )


def load_pipeline(settings: Settings) -> Pipeline:
    if settings.use_stub:
        return StubPipeline(settings)
    from .real_pipeline import RealPipeline  # 延迟导入：stub 模式不加载 torch 等重依赖

    return RealPipeline(settings)
