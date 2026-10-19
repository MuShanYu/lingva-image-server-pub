"""文字行检测 + 文字分割 mask：comic-text-detector（PyTorch 权重 comictextdetector.pt）。

推理流程参照 manga-image-translator 的 detection/ctd.py（ComicTextDetector._infer），
不使用 YOLO 块输出，只用 DB 文字行和分割 mask。
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from ..schemas import BBox
from ..vendor.ctd.basemodel import TextDetBase
from ..vendor.ctd.utils.db_utils import SegDetectorRepresenter
from ..vendor.ctd.utils.imgproc_utils import letterbox

STRIDE = 64
SEG_THRESH = 0.3
LINE_SCORE_MIN = 0.6


@dataclass(frozen=True)
class TextLine:
    pts: np.ndarray  # (4, 2) int，原图坐标
    score: float

    @property
    def bbox(self) -> BBox:
        x1, y1 = self.pts.min(axis=0)
        x2, y2 = self.pts.max(axis=0)
        return (int(x1), int(y1), int(x2), int(y2))

    @property
    def vertical(self) -> bool:
        x1, y1, x2, y2 = self.bbox
        return (y2 - y1) > (x2 - x1)


@dataclass(frozen=True)
class TextDetection:
    lines: list[TextLine]
    mask: np.ndarray  # HxW uint8，文字概率 0~255（未二值化）


class TextDetector:
    def __init__(self, model_path: Path, device: str, detect_size: int) -> None:
        self._device = device
        self._size = max(STRIDE, round(detect_size / STRIDE) * STRIDE)
        # channels_last 让 CPU 上的卷积走更快的 oneDNN 路径，实测 1.6s → 1.2s，结果不变
        self._model = (
            TextDetBase(str(model_path), device=device, act="leaky")
            .to(device)
            .to(memory_format=torch.channels_last)
            .eval()
        )
        self._seg_rep = SegDetectorRepresenter(thresh=SEG_THRESH)

    @torch.no_grad()
    def __call__(self, img: np.ndarray) -> TextDetection:
        h, w = img.shape[:2]
        s = self._size
        padded, _, (dw, dh) = letterbox(img, new_shape=(s, s), auto=False, stride=STRIDE)
        x = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1))[None]).to(self._device)
        x = (x.float() / 255.0).contiguous(memory_format=torch.channels_last)

        _, mask, lines_map = self._model(x)
        # letterbox 只在右侧和下侧补边，裁掉补边后再映射回原图尺寸
        mask = mask[0, 0, : s - dh, : s - dw].float().cpu().numpy()
        lines_map = lines_map[..., : s - dh, : s - dw].float().cpu()

        boxes, scores = self._seg_rep(None, lines_map, height=h, width=w)
        keep = scores[0] > LINE_SCORE_MIN
        lines = [
            TextLine(pts=pts.astype(int), score=float(score))
            for pts, score in zip(boxes[0][keep], scores[0][keep])
        ]
        lines = [ln for ln in lines if ln.bbox[2] - ln.bbox[0] > 1 and ln.bbox[3] - ln.bbox[1] > 1]

        mask_u8 = cv2.resize((mask * 255).clip(0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
        return TextDetection(lines=lines, mask=mask_u8)
