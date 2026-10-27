"""文档识别的模型部分：方向分类、版面分析（PP-DocLayoutV3）、文字检测与识别（PP-OCRv6 + 语种模型）。

只在文档 worker 子进程里导入（requirements-doc.txt）。推理走 PaddleOCR 的 onnxruntime 引擎。
检测和识别分开调用，不用 PaddleOCR 流水线：检测只加载一份，各语种识别模型按需懒加载后常驻；
doc_figure 要先单独跑检测，没有框就提前返回。

显存：onnxruntime 每个会话有自己的显存池，默认只增不减，几个模型的峰值会叠加起来，8G 卡会被占满。
每次推理后收缩显存池（enable_arena_shrinkage），峰值约为单个模型的峰值（notes/doc-ocr-probe.md 第 5 节）。
"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import cv2
import numpy as np

from ..cpus import available_cpus
from .assemble import (
    LayoutBox,
    OcrLine,
    assemble_figure,
    assemble_page,
    filter_detections,
    order_quad,
    to_upright,
    vertical_ratio,
)
from .labels import DEFAULT_REC, DET_MODEL, LAYOUT_MODEL, ORIENT_MODEL, normalize_lang, rec_model_for

logger = logging.getLogger("lingva")

# 方向分类结果 → 把图转正所需的旋转（结果是原图内容被顺时针转过的角度）
_UPRIGHT = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}
# 竖排、高宽比大的行裁出来后转 90° 再识别（与 PaddleOCR 流水线一致）
_ROTATE_CROP_ASPECT = 1.5
# 竖长裁图的识别置信度低于这个值时，转 180° 再试一次（v5 测试集：方向反了的标题 0.20，反过来 1.00）
_FLIP_RETRY_SCORE = 0.6


@dataclass(frozen=True)
class EngineConfig:
    device: str = "gpu:0"  # gpu:N / cpu
    engine: str = "onnxruntime"  # onnxruntime / paddle_static
    det_limit_side: int = 2500  # 检测输入的长边上限
    # 方向分类结果不为 0° 且置信度低于此值时不旋转：竖排日文页会被误判成 90°（置信度 0.83），
    # 正常页和倒置页都在 0.92 以上
    orient_min_score: float = 0.9
    # 方向分类为 90° / 270° 时，原图上竖长检测框的比例达到此值才旋转。分类器对「正文字形侧转、页眉标题横排」
    # 的混排页同样给出 90°（置信度 0.92，与真正横置的页没有区别）；整页横置时几乎所有行都是竖长的，
    # 混排页里还有横排的页眉、标题。不旋转时竖长的行裁出来会转 90° 再识别，侧转的字形照样能认
    rotate_vertical_min: float = 0.8
    rec_batch_size: int = 6  # 识别每批行数；批越大显存峰值越高（64 行约 5.5G）
    figure_min_side: float = 8  # doc_figure：短边小于这个像素数的检测框丢弃
    figure_det_min_score: float = 0.6  # doc_figure：检测分数低于此值的框丢弃
    shrink_arena: bool = True


def enable_arena_shrinkage(device_id: int) -> None:
    """每次 session.run 之后收缩 CUDA 显存池，把这次推理多占的显存还回去。
    paddlex 的 ONNXRuntimeRunner 不支持传 RunOptions，只能包一层 session（paddleocr / paddlex 版本已锁定）"""
    import onnxruntime as ort
    from paddlex.inference.models.runners import onnxruntime_runner as runner

    if getattr(runner.ONNXRuntimeRunner, "_lingva_shrink", False):
        return
    run_options = ort.RunOptions()
    run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", f"gpu:{device_id}")
    create = runner.ONNXRuntimeRunner._create_session

    class _Session:
        def __init__(self, session) -> None:
            self._session = session

        def __getattr__(self, name):
            return getattr(self._session, name)

        def run(self, output_names, feeds):
            return self._session.run(output_names, feeds, run_options)

    runner.ONNXRuntimeRunner._create_session = lambda self: _Session(create(self))
    runner.ONNXRuntimeRunner._lingva_shrink = True


class _Timer:
    def __init__(self) -> None:
        self.ms: dict[str, int] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.ms[name] = self.ms.get(name, 0) + round((time.perf_counter() - t0) * 1000)


def _crop_size(poly: np.ndarray) -> tuple[int, int]:
    pts = poly.astype(np.float32)
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    return max(w, 1), max(h, 1)


def _is_rotated_crop(poly: np.ndarray) -> bool:
    """竖长的行（竖排文字）裁出来后要转 90° 再识别"""
    w, h = _crop_size(poly)
    return h / w >= _ROTATE_CROP_ASPECT


def crop_line(img: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """按检测框透视裁出一行（与 PaddleOCR 的 get_rotate_crop_image 相同）"""
    w, h = _crop_size(poly)
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    m = cv2.getPerspectiveTransform(poly.astype(np.float32), dst)
    crop = cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if _is_rotated_crop(poly):
        crop = np.ascontiguousarray(np.rot90(crop))
    return crop


class DocEngine:
    def __init__(self, cfg: EngineConfig) -> None:
        from paddleocr import DocImgOrientationClassification, LayoutDetection, TextDetection

        self.cfg = cfg
        gpu = cfg.device.startswith("gpu")
        self._device_id = int(cfg.device.partition(":")[2] or 0) if gpu else 0
        if gpu and cfg.engine == "onnxruntime" and cfg.shrink_arena:
            enable_arena_shrinkage(self._device_id)
        self._kwargs: dict = {"device": cfg.device, "engine": cfg.engine}
        if cfg.engine == "onnxruntime":
            # 线程数按容器实际能用的核数（见 app/cpus.py）
            engine_config: dict = {"intra_op_num_threads": available_cpus()}
            if gpu:
                engine_config["provider_options"] = {"device_id": self._device_id, "arena_extend_strategy": "kSameAsRequested"}
            self._kwargs["engine_config"] = engine_config
        self._orient = DocImgOrientationClassification(model_name=ORIENT_MODEL, **self._kwargs)
        self._layout = LayoutDetection(model_name=LAYOUT_MODEL, **self._kwargs)
        self._det = TextDetection(model_name=DET_MODEL, **self._kwargs)
        self._recs: dict[str, object | None] = {}
        self._recognizer(DEFAULT_REC)
        if self._recs[DEFAULT_REC] is None:
            raise RuntimeError(f"默认识别模型 {DEFAULT_REC} 加载失败")

    @property
    def models(self) -> dict:
        return {
            "layout": LAYOUT_MODEL,
            "det": DET_MODEL,
            "orientation": ORIENT_MODEL,
            "rec": sorted(name for name, rec in self._recs.items() if rec is not None),
        }

    def _recognizer(self, model: str):
        """懒加载识别模型；加载失败记一次，之后直接返回 None（调用方降级到默认模型）"""
        if model not in self._recs:
            from paddleocr import TextRecognition

            try:
                self._recs[model] = TextRecognition(model_name=model, **self._kwargs)
                logger.info("识别模型已加载：%s", model)
            except Exception:
                logger.exception("识别模型 %s 加载失败，改用 %s", model, DEFAULT_REC)
                self._recs[model] = None
        return self._recs[model]

    def _rec_for(self, lang_hint: str | None) -> tuple[str, str, object]:
        """返回 (实际语种, 模型名, 识别器)。语种模型不可用时降级到默认模型，语种记为 auto"""
        lang = normalize_lang(lang_hint)
        model = rec_model_for(lang)
        rec = self._recognizer(model)
        if rec is None:
            lang, model, rec = "auto", DEFAULT_REC, self._recs[DEFAULT_REC]
        return lang, model, rec

    def _orientation(self, bgr: np.ndarray) -> int:
        res = next(iter(self._orient.predict(bgr, batch_size=1)))
        angle, score = int(res["label_names"][0]), float(res["scores"][0])
        if angle != 0 and score < self.cfg.orient_min_score:
            logger.info("方向分类 %d°（置信度 %.3f）低于阈值，不旋转", angle, score)
            return 0
        return angle

    def _detect(self, bgr: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
        res = next(iter(self._det.predict(bgr, batch_size=1, limit_side_len=self.cfg.det_limit_side, limit_type="max")))
        return [np.asarray(p) for p in res["dt_polys"]], [float(s) for s in res["dt_scores"]]

    def _recognize(self, rec, bgr: np.ndarray, polys: list[np.ndarray]) -> list[OcrLine]:
        if not polys:
            return []
        crops = [crop_line(bgr, p) for p in polys]
        results = [(str(r["rec_text"]), float(r["rec_score"])) for r in rec.predict(crops, batch_size=self.cfg.rec_batch_size)]
        # 竖长的裁图按竖排文字转了 90°；如果其实是字形侧转的横排文字（如混排页转正后的标题），方向可能正好反了。
        # 置信度低时转 180° 再识别一次，取置信度高的
        retry = [i for i, (_, score) in enumerate(results) if score < _FLIP_RETRY_SCORE and _is_rotated_crop(polys[i])]
        if retry:
            flipped = [np.ascontiguousarray(np.rot90(crops[i], 2)) for i in retry]
            for i, r in zip(retry, rec.predict(flipped, batch_size=self.cfg.rec_batch_size)):
                if float(r["rec_score"]) > results[i][1]:
                    results[i] = (str(r["rec_text"]), float(r["rec_score"]))
        out = []
        for poly, (text, score) in zip(polys, results):
            if text.strip():
                out.append(OcrLine(tuple((float(x), float(y)) for x, y in poly), text, score))
        return out

    def page(self, rgb: np.ndarray, lang_hint: str | None = None) -> dict:
        """doc_page：方向校正 → 版面 → 整页检测 + 识别 → 组装"""
        timer = _Timer()
        lang, model, rec = self._rec_for(lang_hint)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        polys: list[np.ndarray] | None = None
        with timer.stage("orient"):
            angle = self._orientation(bgr)
        if angle in (90, 270):
            # 先在原图上检测，确认整页确实横置；检测框换算到转正后的图上复用，不再检测一次
            with timer.stage("ocr"):
                polys, _ = self._detect(bgr)
            ratio = vertical_ratio([p.tolist() for p in polys])
            if ratio < self.cfg.rotate_vertical_min:
                logger.info("方向分类 %d°，但竖长行只占 %.2f，按混排页处理，不旋转", angle, ratio)
                angle = 0
            else:
                oh, ow = bgr.shape[:2]
                upright = (oh, ow)  # 转正后宽高互换
                polys = [np.asarray(order_quad([to_upright((float(x), float(y)), angle, upright) for x, y in p])) for p in polys]
        if angle:
            with timer.stage("orient"):
                bgr = np.ascontiguousarray(cv2.rotate(bgr, _UPRIGHT[angle]))
        h, w = bgr.shape[:2]
        with timer.stage("layout"):
            res = next(iter(self._layout.predict(bgr, batch_size=1)))
            layout = [LayoutBox(b["label"], tuple(float(v) for v in b["coordinate"]), float(b["score"])) for b in res["boxes"]]
        with timer.stage("ocr"):
            if polys is None:
                polys, _ = self._detect(bgr)
            lines = self._recognize(rec, bgr, polys)
        with timer.stage("assemble"):
            result = assemble_page(lines, layout, angle=angle, upright_size=(w, h))
        logger.info(
            "doc_page %dx%d rotation=%d lang=%s blocks=%d lines=%d timings_ms=%s",
            w, h, angle, lang, len(result["blocks"]), len(lines), timer.ms,
        )
        return {"mode": "doc_page", "lang": lang, "rec_model": model, **result, "timing_ms": timer.ms}

    def figure(self, rgb: np.ndarray, lang_hint: str | None = None) -> dict:
        """doc_figure：不做版面分析和方向分类；先检测，没有框就提前返回"""
        timer = _Timer()
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        h, w = bgr.shape[:2]
        with timer.stage("det"):
            polys, scores = self._detect(bgr)
            keep = filter_detections(
                [p.tolist() for p in polys], scores, min_side=self.cfg.figure_min_side, min_score=self.cfg.figure_det_min_score
            )
            polys = [polys[i] for i in keep]
        lang, model, rec = self._rec_for(lang_hint)
        if not polys:
            logger.info("doc_figure %dx%d 无文字 timings_ms=%s", w, h, timer.ms)
            return {"mode": "doc_figure", "lang": lang, "rec_model": model, "has_text": False, "text_ratio": 0.0,
                    "regions": [], "timing_ms": timer.ms}
        with timer.stage("rec"):
            lines = self._recognize(rec, bgr, polys)
        with timer.stage("assemble"):
            result = assemble_figure(lines, (w, h))
        logger.info(
            "doc_figure %dx%d lang=%s regions=%d lines=%d timings_ms=%s", w, h, lang, len(result["regions"]), len(lines), timer.ms
        )
        return {"mode": "doc_figure", "lang": lang, "rec_model": model, **result, "timing_ms": timer.ms}
