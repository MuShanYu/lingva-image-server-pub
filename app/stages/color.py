"""文字前景色 / 背景色估计，以及彩色字判断。

背景色：块框内、文字 mask（膨胀几像素）以外像素的中位数。
前景色：文字 mask 内与背景色差异最大的 FG_TOP_RATIO 像素的中位数——mask 含抗锯齿边缘和笔画间隙，
直接取中位数会偏向背景色。
"""

import cv2
import numpy as np

from ..schemas import BBox

BG_EXCLUDE_DILATION = 3  # 取背景时先把文字 mask 膨胀几像素，排除描边和抗锯齿边缘
FG_TOP_RATIO = 0.3
DEFAULT_FG = "#000000"
DEFAULT_BG = "#ffffff"
CHROMA_TEXT_MIN = 60  # 笔画像素的色度（RGB 最大通道 - 最小通道）不低于此值时算作彩色
COLORED_STROKE_RATIO = 0.3  # 笔画像素里彩色的占比不低于此值时视为彩色字
CHROMA_PIXEL_MIN = 40  # 单个像素的色度不低于此值时视为「有颜色」（彩色字及其抗锯齿边缘）
PLAIN_BG_COLORED_MAX = 0.2  # 文字以外有颜色的像素占比不超过此值时，视为黑白画面


def chroma(img: np.ndarray) -> np.ndarray:
    """每个像素的色度：RGB 最大通道 - 最小通道（黑白灰为 0）。"""
    return img.max(axis=2).astype(np.int16) - img.min(axis=2)


def is_colored_text(img: np.ndarray, text_mask: np.ndarray) -> bool:
    """text_mask（bool）圈出的文字是不是彩色字（如黑白漫画里的红字）。

    分割 mask 笔画偏粗，含一圈背景：只看其中偏暗的像素（亮度在最暗与最亮的中点以下），它们才是笔画本身。
    不要求全部是彩色：分割图常把字旁边的黑色排线、头发也标成文字。
    """
    px = img[text_mask]
    if len(px) < 20:
        return False
    gray = px.mean(axis=1)
    core = px[gray <= (gray.min() + gray.max()) / 2].astype(np.int16)
    return float((core.max(axis=1) - core.min(axis=1) >= CHROMA_TEXT_MIN).mean()) >= COLORED_STROKE_RATIO


def to_hex(rgb: np.ndarray) -> str:
    r, g, b = (int(v) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def is_plain_background(img: np.ndarray, text_mask: np.ndarray) -> bool:
    """text_mask 以外的画面是不是基本没有颜色（黑白漫画）。彩页上按色度分不出文字和背景。"""
    bg = chroma(img)[~text_mask]
    return len(bg) == 0 or float((bg >= CHROMA_PIXEL_MIN).mean()) <= PLAIN_BG_COLORED_MAX


def estimate_colors(img: np.ndarray, text_mask: np.ndarray, bbox: BBox) -> tuple[str, str]:
    x1, y1, x2, y2 = bbox
    region = img[y1:y2, x1:x2]
    m = text_mask[y1:y2, x1:x2]
    if region.size == 0:
        return DEFAULT_FG, DEFAULT_BG

    k = 2 * BG_EXCLUDE_DILATION + 1
    near_text = cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    fg_px, bg_px = region[m].astype(np.int16), region[~near_text]
    bg_rgb = np.median(bg_px, axis=0) if len(bg_px) else np.array([255, 255, 255])
    if len(fg_px):
        dist = np.abs(fg_px - bg_rgb).max(axis=1)
        fg_px = fg_px[dist >= np.percentile(dist, 100 * (1 - FG_TOP_RATIO))]
    fg = to_hex(np.median(fg_px, axis=0)) if len(fg_px) else DEFAULT_FG
    bg = to_hex(bg_rgb) if len(bg_px) else DEFAULT_BG
    return fg, bg
