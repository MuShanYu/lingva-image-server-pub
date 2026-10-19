import numpy as np
from PIL import Image

from app.real_pipeline import _recognize
from app.stages.group import GroupedBlock, LineIn
from app.stages.ocr import OcrResult

RED = (230, 20, 30)


class FakeOcr:
    """原图裁图（里面有红色像素）一律低置信度；色度图（纯灰度）高置信度。"""

    def __init__(self) -> None:
        self.chroma_seen = 0

    def __call__(self, crops: list[Image.Image]) -> list[OcrResult]:
        out = []
        for c in crops:
            arr = np.asarray(c.convert("RGB"))
            is_chroma = bool((arr[..., 0] == arr[..., 1]).all() and (arr[..., 1] == arr[..., 2]).all())
            self.chroma_seen += is_chroma
            out.append(OcrResult("色度", 0.98) if is_chroma else OcrResult("原图", 0.5))
        return out


def scene(background: tuple[int, int, int]) -> tuple[Image.Image, np.ndarray, GroupedBlock]:
    img = Image.new("RGB", (200, 300), background)
    img.paste((0, 0, 0), (120, 0, 200, 300))  # 压在文字旁边的黑发
    img.paste(RED, (90, 40, 110, 260))
    seg = np.zeros((300, 200), dtype=np.uint8)
    seg[36:264, 86:114] = 255
    bbox = (84, 34, 130, 266)
    return img, seg, GroupedBlock("free", bbox, None, True, [LineIn(bbox, True)], 24)


def test_low_confidence_colored_text_is_retried_on_chroma_image() -> None:
    img, seg, blk = scene((255, 255, 255))
    ocr = FakeOcr()
    [res] = _recognize(ocr, img, seg, [blk])
    assert res.text == "色度" and ocr.chroma_seen == 1


def test_chroma_retry_is_skipped_on_colored_pages() -> None:
    img, seg, blk = scene((250, 200, 80))  # 彩页：背景本身有颜色，色度图分不出文字
    ocr = FakeOcr()
    [res] = _recognize(ocr, img, seg, [blk])
    assert res.text == "原图" and ocr.chroma_seen == 0
