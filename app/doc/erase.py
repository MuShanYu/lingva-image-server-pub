"""文档擦字（doc_page、doc_figure）：行多边形按行粗细膨胀成 mask，交给漫画管线的 inpaint_image——mask 外一圈是纯色
（图表、截图大多如此）时直接用中位色填充，其余区域用常驻的 LaMa 修补，不另外加载模型。"""

from collections.abc import Sequence

import cv2
import numpy as np

DILATE_RATIO = 0.15  # 膨胀行粗细（多边形短边）的这个比例，盖住字形边缘的抗锯齿


def _thickness(pts: np.ndarray) -> float:
    a = (np.linalg.norm(pts[0] - pts[1]) + np.linalg.norm(pts[2] - pts[3])) / 2
    b = (np.linalg.norm(pts[1] - pts[2]) + np.linalg.norm(pts[3] - pts[0])) / 2
    return float(min(a, b))


def text_mask(height: int, width: int, polys: Sequence[Sequence[Sequence[float]]]) -> np.ndarray:
    """行多边形（像素坐标）→ uint8 mask（文字处 255）。每行按自己的粗细膨胀，只在行附近的小块区域里做"""
    mask = np.zeros((height, width), np.uint8)
    for poly in polys:
        pts = np.asarray(poly, np.float32)
        if len(pts) < 3:
            continue
        k = max(1, round(DILATE_RATIO * (_thickness(pts) if len(pts) == 4 else 0)))
        x1, y1 = np.floor(pts.min(axis=0)).astype(int) - k - 1
        x2, y2 = np.ceil(pts.max(axis=0)).astype(int) + k + 2
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = np.zeros((y2 - y1, x2 - x1), np.uint8)
        cv2.fillPoly(roi, [np.round(pts - (x1, y1)).astype(np.int32)], 255)
        roi = cv2.dilate(roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
        np.maximum(mask[y1:y2, x1:x2], roi, out=mask[y1:y2, x1:x2])
    return mask
