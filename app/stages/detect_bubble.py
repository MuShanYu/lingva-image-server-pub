"""气泡 / 文字区域检测：ogkalu/comic-text-and-bubble-detector（RT-DETRv2，detector.onnx 完整版）。

输入输出格式（核实来源：HF 仓库 commit 16e8a622 的 config.json / preprocessor_config.json，
以及 onnxruntime 读取 detector.onnx 的输入输出签名）：
- 输入 images: float32 [N,3,H,W]，RGB，直接拉伸到 640x640（resample=2 双线性），/255，不做 mean/std 归一化
- 输入 orig_target_sizes: int64 [N,2] = [width, height]
- 输出 labels [N,300]、boxes [N,300,4]（xyxy，原图像素坐标）、scores [N,300]
- 类别（config.json id2label）：0 = bubble，1 = text_bubble，2 = text_free
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from ..schemas import BBox

INPUT_SIZE = 640
NMS_IOU = 0.6

BUBBLE = 0
TEXT_BUBBLE = 1
TEXT_FREE = 2
LABEL_NAMES = {BUBBLE: "bubble", TEXT_BUBBLE: "text_bubble", TEXT_FREE: "text_free"}


@dataclass(frozen=True)
class Region:
    bbox: BBox
    label: int
    score: float


def _iou(a: BBox, b: BBox) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(regions: list[Region], iou_thresh: float = NMS_IOU) -> list[Region]:
    """同类别内按分数做 NMS。RT-DETR 本身无需 NMS，这里只兜底去掉偶发的重复框。"""
    kept: list[Region] = []
    for r in sorted(regions, key=lambda r: r.score, reverse=True):
        if all(k.label != r.label or _iou(k.bbox, r.bbox) < iou_thresh for k in kept):
            kept.append(r)
    return kept


class BubbleDetector:
    def __init__(self, model_path: Path, device: str, score_min: float) -> None:
        providers: list = ["CPUExecutionProvider"]
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            # onnxruntime-gpu 自己找不到 pip 装的 CUDA / cuDNN 库（torch 自带的那份），先按 nvidia-* 包的位置加载
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            # 显存按需申请，不按 2 的幂预留，给 torch 的模型多留些显存
            providers.insert(0, ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}))
        opts = ort.SessionOptions()
        # 否则 onnxruntime 会清除线程的 denormal flush 标志，拖慢后续的 torch 推理（见 device.flush_denormal）
        opts.add_session_config_entry("session.set_denormal_as_zero", "1")
        self._session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
        self.provider = self._session.get_providers()[0]
        self._score_min = score_min

    def __call__(self, img: np.ndarray) -> list[Region]:
        h, w = img.shape[:2]
        x = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        sizes = np.array([[w, h]], dtype=np.int64)
        labels, boxes, scores = self._session.run(None, {"images": x, "orig_target_sizes": sizes})

        regions: list[Region] = []
        for label, box, score in zip(labels[0], boxes[0], scores[0]):
            if score < self._score_min or int(label) not in LABEL_NAMES:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            bbox = (
                max(0, int(np.floor(x1))),
                max(0, int(np.floor(y1))),
                min(w, int(np.ceil(x2))),
                min(h, int(np.ceil(y2))),
            )
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                regions.append(Region(bbox=bbox, label=int(label), score=float(score)))
        return nms(regions)
