"""擦字 mask 生成（numpy / OpenCV，无模型）。

1. CTD 分割概率图二值化；
2. 只保留落在「块框外扩 BLOCK_MARGIN_RATIO × 字号」范围内的部分，避免把画面线条当成文字。
   范围取整个块而不是各行框：CTD 行框常常漏掉行首的「○」、行尾的「？」，或整行漏检，
   而这些字在分割图里是有的；
   兜底块（CTD 没有检出行，见 group.py）的分割结果不一定完整，再加上 Otsu 在区域内分出的文字像素；
   黑白画面上的彩色字（如红字）：只保留有颜色的像素附近的部分。分割图会把字旁边的排线、头发、
   五官标成文字，Otsu 会把黑发和涂黑算成文字，它们都没有颜色，不去掉的话会把画面擦坏；
3. 膨胀覆盖描边和毛边：膨胀量按块字号的 DILATION_RATIO 计算，至少 DILATION_MIN，最多 MASK_DILATION；
   固定的大膨胀量（如 30px）会吃掉离文字很近的气泡边框和旁白框线；
4. bubble 块的 mask 限制在气泡框内；
5. 保护线稿：离文字笔画超过 STROKE_KEEP 像素的深色像素（气泡边框、框线、画面线条）从 mask 中扣除，
   不会被擦掉；膨胀出来的部分只擦浅色像素（描边、白边）。
   彩色字（如红字）的灰度也很低，但它不是线稿：彩色字块的 mask 里有颜色的像素照常擦除，
   否则笔画边缘会留下一圈彩色残边，还会让擦字时把白底误判成「线稿被截断」而改用 LaMa。
"""

from collections.abc import Sequence
from dataclasses import replace

import cv2
import numpy as np

from ..schemas import BBox
from .color import CHROMA_PIXEL_MIN, chroma, is_colored_text, is_plain_background
from .group import GroupedBlock, union_bbox

TEXT_THRESH = 127  # CTD 分割概率（0~255）二值化阈值
LINE_MARGIN_RATIO = 0.25  # 兜底块的区域外扩比例（相对块字号）
LINE_MARGIN_MIN = 4
BLOCK_MARGIN_RATIO = 0.6  # 块框外扩比例（相对块字号），覆盖行框外的「○」「？」等
DILATION_RATIO = 0.2  # 膨胀量相对块字号的比例
DILATION_MIN = 3
LINE_ART_THRESH = 160  # 灰度低于此值视为线稿（细线抗锯齿后常在 100~150）
STROKE_KEEP = 3  # 文字笔画外这么多像素内的深色像素仍视为文字（抗锯齿边缘），可以擦除
REGION_EXTEND_MIN = 0.15  # 文字区域里的分割文字有这么大比例落在块框外时，把块框扩到区域


def _clip(box: BBox, w: int, h: int) -> BBox:
    return (max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3]))


def _expand(box: BBox, m: int, w: int, h: int) -> BBox:
    return _clip((box[0] - m, box[1] - m, box[2] + m, box[3] + m), w, h)


def otsu_text_mask(gray: np.ndarray) -> np.ndarray:
    """区域内 Otsu 二值化，取像素较少的一类作为文字（文字笔画通常比背景面积小）。"""
    if gray.size == 0:
        return np.zeros_like(gray, dtype=bool)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = binary == 0
    return dark if dark.sum() <= dark.size / 2 else ~dark


def _line_zones(blk: GroupedBlock, w: int, h: int) -> list[BBox]:
    margin = max(LINE_MARGIN_MIN, round(LINE_MARGIN_RATIO * blk.font_size))
    return [_expand(ln.bbox, margin, w, h) for ln in blk.lines]


def _block_zone(blk: GroupedBlock, w: int, h: int) -> BBox:
    """块取分割文字的范围：块框外扩 BLOCK_MARGIN_RATIO × 字号。"""
    return _expand(blk.bbox, max(LINE_MARGIN_MIN, round(BLOCK_MARGIN_RATIO * blk.font_size)), w, h)


def extend_to_regions(seg: np.ndarray, blocks: Sequence[GroupedBlock]) -> list[GroupedBlock]:
    """CTD 行框只盖住了一段文字的前半截时（后半截在分割图里有、也在文字区域里），把块框扩到区域。

    不扩的话后半截既不会送去 OCR，也不会被擦掉。只看分割图里有没有落在外面的文字，不直接比较
    行框和区域框的大小：区域框普遍比行框松一圈，大小差异说明不了问题。
    """
    h, w = seg.shape[:2]
    text = seg > TEXT_THRESH
    out: list[GroupedBlock] = []
    for blk in blocks:
        if blk.region_bbox is not None and not blk.from_region_only:
            rx1, ry1, rx2, ry2 = _clip(blk.region_bbox, w, h)
            bx1, by1, bx2, by2 = blk.bbox
            total = int(text[ry1:ry2, rx1:rx2].sum())
            inside = int(text[max(ry1, by1) : min(ry2, by2), max(rx1, bx1) : min(rx2, bx2)].sum())
            if total > 0 and total - inside >= REGION_EXTEND_MIN * total:
                blk = replace(blk, bbox=union_bbox([blk.bbox, (rx1, ry1, rx2, ry2)]))
        out.append(blk)
    return out


def block_text_masks(
    img: np.ndarray, seg: np.ndarray, blocks: Sequence[GroupedBlock]
) -> list[np.ndarray]:
    """每个块各自的文字笔画 mask（HxW bool，未膨胀），用于擦字和颜色估计。"""
    h, w = seg.shape[:2]
    text = seg > TEXT_THRESH
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    masks: list[np.ndarray] = []
    for blk in blocks:
        m = np.zeros((h, w), dtype=bool)
        zones = _line_zones(blk, w, h) if blk.from_region_only else [_block_zone(blk, w, h)]
        for x1, y1, x2, y2 in zones:
            zone, zm = img[y1:y2, x1:x2], text[y1:y2, x1:x2]
            colored_text = is_colored_text(zone, zm)
            if blk.from_region_only:
                zm = zm | otsu_text_mask(gray[y1:y2, x1:x2])
            if colored_text and is_plain_background(zone, zm):
                colored = (chroma(zone) >= CHROMA_PIXEL_MIN).astype(np.uint8)
                zm = zm & (cv2.dilate(colored, _ellipse(STROKE_KEEP)) > 0)
            m[y1:y2, x1:x2] |= zm
        masks.append(m)
    return masks


def _ellipse(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def block_dilation(font_size: int, max_dilation: int) -> int:
    return max(0, min(max_dilation, max(DILATION_MIN, round(DILATION_RATIO * font_size))))


def erase_mask(
    img: np.ndarray, text_masks: Sequence[np.ndarray], blocks: Sequence[GroupedBlock], max_dilation: int
) -> np.ndarray:
    """合并各块文字 mask 并膨胀，bubble 块限制在气泡框内，并扣除线稿。返回 HxW uint8（0 / 255）。"""
    h, w = img.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    strokes = np.zeros((h, w), dtype=np.uint8)
    colored = chroma(img) >= CHROMA_PIXEL_MIN
    colored_text = np.zeros((h, w), dtype=bool)  # 彩色字块 mask 里有颜色的像素：是文字，不是线稿
    for m, blk in zip(text_masks, blocks):
        mu8 = m.astype(np.uint8) * 255
        strokes |= mu8
        d = block_dilation(blk.font_size, max_dilation)
        if d > 0:
            mu8 = cv2.dilate(mu8, _ellipse(d))
        if blk.kind == "bubble" and blk.bubble_bbox is not None:
            bx1, by1, bx2, by2 = _clip(blk.bubble_bbox, w, h)
            limit = np.zeros_like(mu8)
            limit[by1:by2, bx1:bx2] = 255
            mu8 &= limit
        out |= mu8
        if is_colored_text(img, m):
            colored_text |= (mu8 > 0) & colored

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    near_text = cv2.dilate(strokes, _ellipse(STROKE_KEEP)) > 0
    out[(gray < LINE_ART_THRESH) & ~near_text & ~colored_text] = 0
    return out


RESIDUE_ZONE = 6  # 检查范围：擦字 mask 外扩的像素数（「○」的残边、问号的点就落在 mask 边上）
RESIDUE_GUARD = 3  # 检查范围外再加一圈保护带：连通块伸进保护带说明是延伸出去的线条，不是残留
RESIDUE_DELTA = 45  # 与局部背景的灰度差（擦字底图和原图上都要超过）
RESIDUE_SEG_THRESH = 64  # 原图 CTD 分割概率的较低阈值，只要有一点「像文字」就算
RESIDUE_SEG_DILATE = 3
RESIDUE_GROW = 2  # 补擦 mask 在残留像素外再扩的像素数


def residue_mask(original: np.ndarray, clean: np.ndarray, mask: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """擦字后残留的零碎笔画：行框外没擦到的「○」边缘、问号的点、LaMa 留下的灰色小团。

    一个像素算残留，需同时满足：落在擦字 mask 外扩 RESIDUE_ZONE 的范围内；擦字底图和原图上都与
    局部背景（该范围内底图的灰度中位数）相差超过 RESIDUE_DELTA；原图 CTD 分割图认为它附近是文字
    （排除气泡边框和画面线条）；所在连通块完全落在检查范围内（伸出去的是长线条）。
    返回 HxW uint8（0 / 255），没有残留时全 0。
    """
    m = mask > 0
    zone = cv2.dilate(m.astype(np.uint8), _ellipse(RESIDUE_ZONE)) > 0
    outer = cv2.dilate(m.astype(np.uint8), _ellipse(RESIDUE_ZONE + RESIDUE_GUARD)) > 0
    g_clean = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY).astype(np.int16)
    g_orig = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.int16)
    textish = cv2.dilate((seg > RESIDUE_SEG_THRESH).astype(np.uint8), _ellipse(RESIDUE_SEG_DILATE)) > 0

    n, labels = cv2.connectedComponents(outer.astype(np.uint8), connectivity=8)
    bg = np.zeros(n, dtype=np.int16)
    for i in range(1, n):
        bg[i] = int(np.median(g_clean[(labels == i) & zone]))
    local_bg = bg[labels]
    cand = (
        outer
        & (np.abs(g_clean - local_bg) > RESIDUE_DELTA)
        & (np.abs(g_orig - local_bg) > RESIDUE_DELTA)
        & textish
    )

    out = np.zeros(mask.shape, dtype=np.uint8)
    cn, clabels, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    outside = outer & ~zone
    for i in range(1, cn):
        x, y, w, h, _ = stats[i]
        comp = clabels[y : y + h, x : x + w] == i
        if (comp & outside[y : y + h, x : x + w]).any():
            continue
        out[y : y + h, x : x + w][comp] = 255
    if out.any():
        out = cv2.dilate(out, _ellipse(RESIDUE_GROW))
    return out
