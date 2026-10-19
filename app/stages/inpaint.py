"""擦字：按区域修补，而不是把整页缩放后一次性修补。

1. 逐个连通区域判断：mask 外一圈几乎是纯色（白气泡、白色旁白框）时直接用中位色填充，
   跳过 LaMa——更干净，CPU 上也快得多；
2. 其余区域中相距 ≤ REGION_MERGE_GAP 的合并，四周外扩上下文（区域尺寸的 CONTEXT_RATIO，
   至少 CONTEXT_MIN 像素）后裁剪；
3. 交给 LaMa large：最长边超过 INPAINT_SIZE 时先缩小，边长反射补边到 8 的倍数；
4. 只在 mask 内贴回，边缘高斯羽化，避免接缝。

LaMa 推理方式参照 manga-image-translator 的 inpainting_lama_mpe.py：输入 [0,1]、mask 区域置零；
权重保持 fp32，GPU 上只用 autocast（权重直接转 fp16/bf16 会让结果发暗，fp16 autocast 同样不可用）。
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from ..schemas import BBox
from ..vendor.lama.lama import load_lama_mpe
from .mask import LINE_ART_THRESH

REGION_MERGE_GAP = 32
CONTEXT_RATIO = 0.5
CONTEXT_MIN = 64
PAD_MULTIPLE = 8
# GPU 上补边到 64 的倍数：LaMa 的 FFC 层按输入尺寸建 cuFFT 计划、cuDNN 也按尺寸缓存，区域尺寸五花八门时
# 几乎每次都是新尺寸，单次调用多出约 40ms 固定开销；对齐到 64 后尺寸种类有限，缓存很快命中。
# CPU 上计算量随面积增长，仍按 8 对齐
PAD_MULTIPLE_CUDA = 64
FEATHER_SIGMA = 1.5  # 羽化约 2~3 像素
FLAT_RING = 6  # 纯色判断：取 mask 外这么宽的一圈像素
FLAT_TOL = 16  # 与中位色每个通道相差不超过此值的像素视为「同色」
FLAT_RATIO_MIN = 0.9  # 同色像素占比不低于此值时视为纯色
FLAT_LIGHT_MIN = 0.6  # 外圈浅色像素占比不低于此值时，视为浅色背景，判断时排除深色线稿
LINE_ENTER_MIN = 3  # 外圈深色线条与 mask 内深色像素相接超过这么多像素时，视为线稿被 mask 截断


def _gap(a: int, b: int, c: int, d: int) -> int:
    return max(0, max(a, c) - min(b, d))


def _close(a: BBox, b: BBox, gap: int) -> bool:
    return _gap(a[0], a[2], b[0], b[2]) <= gap and _gap(a[1], a[3], b[1], b[3]) <= gap


def plan_regions(mask: np.ndarray, gap: int = REGION_MERGE_GAP) -> list[BBox]:
    """mask 连通区域的外接框，相近的框反复合并直到稳定。"""
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    boxes: list[BBox] = [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h, _ in stats[1:n]]
    changed = True
    while changed:
        changed = False
        merged: list[BBox] = []
        for box in boxes:
            for i, m in enumerate(merged):
                if _close(box, m, gap):
                    merged[i] = (min(box[0], m[0]), min(box[1], m[1]), max(box[2], m[2]), max(box[3], m[3]))
                    changed = True
                    break
            else:
                merged.append(box)
        boxes = merged
    return boxes


def context_box(box: BBox, w: int, h: int) -> BBox:
    x1, y1, x2, y2 = box
    mx = max(CONTEXT_MIN, round(CONTEXT_RATIO * (x2 - x1)))
    my = max(CONTEXT_MIN, round(CONTEXT_RATIO * (y2 - y1)))
    return (max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my))


def flat_fill_color(crop: np.ndarray, m: np.ndarray) -> np.ndarray | None:
    """m（bool）外一圈几乎是纯色时返回其中位色，否则返回 None。

    外圈以浅色为主时，先排除其中的深色线稿（气泡边框、框线）：mask 已经扣掉了线稿（见 mask.py），
    填色不会盖到它们，不应该因为外圈穿过边框就判成「非纯色」；浅灰的网点纹理仍参与判断。
    """
    k = 2 * FLAT_RING + 1
    ring = cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool) & ~m
    px = crop[ring].astype(np.int16)
    if len(px) < 20:
        return None
    crop_dark = crop.mean(axis=2) < LINE_ART_THRESH
    # 外圈的深色线条一直延伸进 mask（离文字太近的气泡边框被划进了 mask）：纯色填充会把这段线
    # 涂掉，留给 LaMa，它能把线补回来
    ring_dark = (ring & crop_dark).astype(np.uint8)
    if (cv2.dilate(ring_dark, np.ones((3, 3), np.uint8)).astype(bool) & m & crop_dark).sum() >= LINE_ENTER_MIN:
        return None
    gray = px.mean(axis=1)
    light = gray >= LINE_ART_THRESH
    if light.mean() >= FLAT_LIGHT_MIN:
        px = px[light]
    median = np.median(px, axis=0)
    same = (np.abs(px - median).max(axis=1) <= FLAT_TOL).mean()
    return median.astype(np.uint8) if same >= FLAT_RATIO_MIN else None


def fill_flat_regions(out: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int]:
    """把背景为纯色的连通区域直接填色（原地修改 out），返回剩余需要 LaMa 的 mask 和填充个数。"""
    h, w = mask.shape
    rest = mask.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    filled = 0
    pad = FLAT_RING + 1
    for i in range(1, n):
        x, y, cw, ch, _ = stats[i]
        x1, y1, x2, y2 = max(0, x - pad), max(0, y - pad), min(w, x + cw + pad), min(h, y + ch + pad)
        comp = labels[y1:y2, x1:x2] == i
        color = flat_fill_color(out[y1:y2, x1:x2], comp)
        if color is None:
            continue
        out[y1:y2, x1:x2][comp] = color
        rest[y1:y2, x1:x2][comp] = 0
        filled += 1
    return rest, filled


class LamaInpainter:
    def __init__(self, model_path: Path, device: str, autocast_dtype: torch.dtype | None, max_size: int) -> None:
        self._device = device
        self._dtype = autocast_dtype
        self._max_size = max_size
        self._pad = PAD_MULTIPLE_CUDA if device == "cuda" else PAD_MULTIPLE
        self._model = load_lama_mpe(str(model_path), device=device, use_mpe=False, large_arch=True)

    @torch.no_grad()
    def __call__(self, crop: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        img, mask = crop, crop_mask
        scale = min(1.0, self._max_size / max(h, w))
        if scale < 1.0:
            size = (max(1, round(w * scale)), max(1, round(h * scale)))
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
        sh, sw = img.shape[:2]
        ph, pw = -sh % self._pad, -sw % self._pad
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT)
        mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)

        x = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).to(self._device)
        m = (torch.from_numpy(mask)[None, None].float() / 255 >= 0.5).float().to(self._device)
        x = x * (1 - m)
        if self._dtype is not None:
            with torch.autocast(device_type="cuda", dtype=self._dtype):
                out = self._model(x, m)
        else:
            out = self._model(x, m)
        out_np = (out.float()[0].permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
        out_np = out_np[:sh, :sw]
        if scale < 1.0:
            out_np = cv2.resize(out_np, (w, h), interpolation=cv2.INTER_CUBIC)
        return out_np


@dataclass(frozen=True)
class InpaintStats:
    regions: int
    lama: int
    flat: int


def inpaint_image(img: np.ndarray, mask: np.ndarray, lama: LamaInpainter) -> tuple[np.ndarray, InpaintStats]:
    h, w = img.shape[:2]
    out = img.copy()
    rest, n_flat = fill_flat_regions(out, mask)
    regions = plan_regions(rest)
    for box in regions:
        x1, y1, x2, y2 = context_box(box, w, h)
        crop, cm = out[y1:y2, x1:x2], rest[y1:y2, x1:x2]
        filled = lama(crop, cm)
        m01 = (cm > 0).astype(np.float32)
        alpha = np.maximum(m01, cv2.GaussianBlur(m01, (0, 0), FEATHER_SIGMA))[..., None]
        out[y1:y2, x1:x2] = (filled * alpha + crop * (1 - alpha)).round().astype(np.uint8)
    return out, InpaintStats(n_flat + len(regions), len(regions), n_flat)
