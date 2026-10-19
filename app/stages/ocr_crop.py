"""OCR 输入的裁图方案。

manga-ocr 把输入缩放成 224×224：竖排多列的块整块识别效果很好，但很长的横排行会被压扁而「编造」文字。
- 竖排块、兜底块（没有可靠的行 / 分割结果）：整块裁一张；
- 横排块：逐行裁图，行的宽高比超过 MAX_ASPECT 时在字间空隙处切成几段，识别后按顺序拼接。
  行不用 CTD 的行框（它偶尔一个框跨好几行、或与别的行框重叠，逐行识别会读出重复内容），
  而是在块内对文字笔画做水平投影切出文字行；投影里切不出行时才退回用 CTD 行框。
  笔画 = 块内 Otsu 二值化的文字像素 ∩ 膨胀后的 CTD 分割图：CTD 分割图笔画很粗、行间几乎
  没有空隙，Otsu 的结果行间空隙清楚但可能混进画面纹理，二者取交集。
每张裁图里，其他块的文字行都涂白，避免把邻近的字读进来。
裁图边距和切段宽高比由 CropParams 决定：低置信度的块会换 RETRY_PARAMS 里的几组参数重新识别，
取置信度最高的结果（见 real_pipeline._recognize）。
彩色字（如压在黑发、网点上的红字）转成灰度后和背景几乎没有对比度，manga-ocr 读不出来：
低置信度的彩色字块再用 CHROMA_PARAMS 识别一次，裁图换成「色度图」——有颜色的像素变黑、
黑白灰的画面全部变白，只剩下文字。
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..schemas import BBox
from .color import chroma
from .group import GroupedBlock, intersection, ioa
from .mask import otsu_text_mask

CROP_MARGIN = 4
MAX_ASPECT = 5.0  # 实测横排长行按 ≤5:1 切段后识别准确率明显提升，再细容易切到字中间
TEXT_THRESH = 127
ROW_HEIGHT_RATIO = 0.8  # 字形实际高度约为块字号（DB 行框粗细，含外扩）的这个比例，作为行高初值
ROW_GAP_MERGE = 0.15  # 行内小于「行高 × 此值」的空白视为同一行（例如「一」上下的空白）
ROW_MIN = 0.3  # 高度低于「行高 × 此值」的横条视为噪点（框线、点）
ROW_SPLIT = 1.6  # 高于「实际行高中位数 × 此值」的横条视为几行粘在一起，按行高均分
SEG_DILATE = 3  # CTD 分割图膨胀像素，用来圈定 Otsu 结果里哪些是文字
CHROMA_GAIN = 1.5  # 色度图的对比度：色度 × 此值后取反，纯红（色度 255）和偏淡的彩色字都能变成深色


@dataclass(frozen=True)
class CropParams:
    margin: int = CROP_MARGIN
    max_aspect: float = MAX_ASPECT
    chroma: bool = False  # 裁图换成色度图（只用于彩色字）


DEFAULT_PARAMS = CropParams()
CHROMA_PARAMS = CropParams(chroma=True)
# 重试用：多留上下文 + 切段略长；贴边裁 + 切段略短
RETRY_PARAMS = (CropParams(margin=12, max_aspect=6.0), CropParams(margin=1, max_aspect=4.0))


def _expand(box: BBox, w: int, h: int, m: int = CROP_MARGIN) -> BBox:
    return (max(0, box[0] - m), max(0, box[1] - m), min(w, box[2] + m), min(h, box[3] + m))


def masked_crop(
    img: Image.Image, box: BBox, own: Sequence[BBox], others: Sequence[BBox], margin: int = CROP_MARGIN
) -> Image.Image:
    """裁出 box（外扩 margin 边距），把与之相交的 others 涂白，再贴回 own 的原图内容。"""
    crop_box = _expand(box, img.width, img.height, margin)
    crop = img.crop(crop_box)
    foreign = [o for o in others if intersection(o, crop_box) > 0]
    if not foreign:
        return crop
    ox, oy = crop_box[0], crop_box[1]
    draw = ImageDraw.Draw(crop)
    for x1, y1, x2, y2 in foreign:
        draw.rectangle((x1 - ox, y1 - oy, x2 - ox, y2 - oy), fill="white")
    for b in own:
        crop.paste(img.crop(b), (b[0] - ox, b[1] - oy))
    return crop


def chroma_image(crop: Image.Image) -> Image.Image:
    """色度图：有颜色的像素变深，黑白灰变白。"""
    c = chroma(np.asarray(crop.convert("RGB")))
    return Image.fromarray((255 - np.clip(c * CHROMA_GAIN, 0, 255)).astype(np.uint8)).convert("RGB")


def split_points(ink: np.ndarray, pieces: int, window: int) -> list[int]:
    """在 ink（每列的文字像素数）上找 pieces-1 个切点：各自在等分位置 ±window/2 内取墨迹最少的列。"""
    n = len(ink)
    cuts: list[int] = []
    for k in range(1, pieces):
        target = round(n * k / pieces)
        lo, hi = max(1, target - window // 2), min(n - 1, target + window // 2 + 1)
        cuts.append(lo + int(np.argmin(ink[lo:hi])) if hi > lo else target)
    return cuts


def split_long_line(crop: Image.Image, seg_crop: np.ndarray, max_aspect: float = MAX_ASPECT) -> list[Image.Image]:
    """横排行过长时按字间空隙切成宽高比约 ≤ max_aspect 的几段。"""
    w, h = crop.size
    pieces = max(1, math.ceil(w / (h * max_aspect)))
    if pieces == 1:
        return [crop]
    ink = (seg_crop > TEXT_THRESH).sum(axis=0).astype(np.float32)
    ink = np.convolve(ink, np.ones(3, dtype=np.float32) / 3, mode="same")
    cuts = [0, *split_points(ink, pieces, h), w]
    return [crop.crop((a, 0, b, h)) for a, b in zip(cuts, cuts[1:]) if b > a]


def text_rows(text: np.ndarray, row_height: float) -> list[tuple[int, int]]:
    """在文字 mask（bool，HxW）上按水平投影切出文字行，返回 [(y1, y2), ...]（y2 不含）。"""
    has_ink = text.any(axis=1)
    runs: list[list[int]] = []
    y = 0
    while y < len(has_ink):
        if has_ink[y]:
            start = y
            while y < len(has_ink) and has_ink[y]:
                y += 1
            if runs and start - runs[-1][1] < ROW_GAP_MERGE * row_height:
                runs[-1][1] = y
            else:
                runs.append([start, y])
        else:
            y += 1

    kept = [(y1, y2) for y1, y2 in runs if y2 - y1 >= ROW_MIN * row_height]
    # 行高以实际切出来的行为准（字号估计含 DB 外扩，偏大）；只有一段时没法比较，用初值
    ref = float(np.median([y2 - y1 for y1, y2 in kept])) if len(kept) >= 2 else row_height
    rows: list[tuple[int, int]] = []
    for y1, y2 in kept:
        h = y2 - y1
        parts = round(h / ref) if h > ROW_SPLIT * ref else 1
        step = h / parts
        rows += [(y1 + round(step * k), y1 + round(step * (k + 1))) for k in range(parts)]
    return rows


def ocr_crops(
    img: Image.Image,
    seg: np.ndarray,
    block: GroupedBlock,
    blocks: Sequence[GroupedBlock],
    params: CropParams = DEFAULT_PARAMS,
) -> list[Image.Image]:
    """一个块送进 OCR 的裁图列表，按阅读顺序排列（识别结果直接拼接即可）。"""
    # 大部分落在本块里的「别的块的行」多半是分组误差，不涂白，免得把本块的字擦掉
    other_lines = [
        ln.bbox for o in blocks if o is not block for ln in o.lines if ioa(ln.bbox, block.bbox) < 0.5
    ]
    own_lines = [ln.bbox for ln in block.lines]
    if block.vertical or block.from_region_only:
        crops = [masked_crop(img, block.bbox, own_lines, other_lines, params.margin)]
        return [chroma_image(c) for c in crops] if params.chroma else crops

    bx1, by1, bx2, by2 = _expand(block.bbox, img.width, img.height)
    seg_near = cv2.dilate((seg[by1:by2, bx1:bx2] > TEXT_THRESH).astype(np.uint8), np.ones((2 * SEG_DILATE + 1,) * 2, np.uint8))
    gray = cv2.cvtColor(np.asarray(img.crop((bx1, by1, bx2, by2))), cv2.COLOR_RGB2GRAY)
    text = otsu_text_mask(gray) & (seg_near > 0)
    rows = text_rows(text, ROW_HEIGHT_RATIO * block.font_size)
    if rows:
        boxes = []
        for ry1, ry2 in rows:
            cols = np.flatnonzero(text[ry1:ry2].any(axis=0))
            boxes.append((bx1 + int(cols[0]), by1 + ry1, bx1 + int(cols[-1]) + 1, by1 + ry2))
    else:
        boxes = own_lines

    crops: list[Image.Image] = []
    for box in boxes:
        crop = masked_crop(img, box, [], other_lines, params.margin)
        x1, y1, x2, y2 = _expand(box, img.width, img.height, params.margin)
        crops += split_long_line(crop, seg[y1:y2, x1:x2], params.max_aspect)
    return [chroma_image(c) for c in crops] if params.chroma else crops
