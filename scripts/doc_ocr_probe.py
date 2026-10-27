"""文档识别管线的探针：确认实际安装的 PaddleOCR 版本的接口、结果结构、耗时和显存（design/文档识别管线-实现说明.md 第 2 节）。

用法：
  python scripts/doc_ocr_probe.py <图片目录 | PDF 文件 | PDF 目录> --out probe-out [--engine onnxruntime]
      [--sides 2000,2500,3000] [--pages 6] [--with-comic] [--lang-pages ko=06_korean,ru=07_russian]
      [--ort-shrink] [--ort-opt cudnn_conv_algo_search=HEURISTIC ...] [--skip-structure] [--skip-lang]

结论见 notes/doc-ocr-probe.md。在 GPU 机上测显存时建议加 --ort-shrink（每次推理后收缩显存池，否则各模型的峰值叠加）。
PDF 没有嵌入 CJK 字体时，渲染环境必须装有 CJK 字体（如 fonts-noto-cjk），否则正文是空白。

PDF 按 200 DPI 渲染（长边不超过 2500px），只在内存里。结论写到 <out>/probe.json；
<out> 下的可视化图和 PaddleOCR 原始 JSON 含识别文字，只用于人工查看，不要提交。
--with-comic 同时加载漫画流水线（读 .env / 环境变量，需 USE_STUB=0），测两者共存时的显存。
"""

import argparse
import inspect
import json
import logging
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
RENDER_DPI = 200
RENDER_MAX_SIDE = 2500

LAYOUT_MODEL = "PP-DocLayoutV3"
DET_MODEL = "PP-OCRv6_medium_det"
REC_MODEL = "PP-OCRv6_medium_rec"
# 语种识别模型的候选名：以 paddlex 官方模型列表为准，探针会把列表里匹配的名字都列出来
LANG_PREFIXES = {"ko": ("korean",), "ru": ("eslav", "cyrillic", "ru"), "th": ("th",)}

logger = logging.getLogger("probe")


# ---------- 输入 ----------


def _limit(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def _render_pdf(path: Path) -> list[tuple[str, np.ndarray]]:
    import pypdfium2 as pdfium

    pages = []
    pdf = pdfium.PdfDocument(str(path))
    try:
        for i, page in enumerate(pdf):
            bitmap = page.render(scale=RENDER_DPI / 72)
            img = np.asarray(bitmap.to_pil().convert("RGB"))
            pages.append((f"{path.stem}#p{i + 1}", _limit(img, RENDER_MAX_SIDE)))
    finally:
        pdf.close()
    return pages


def load_pages(src: Path) -> list[tuple[str, np.ndarray]]:
    """返回 [(名字, RGB 图)]，全部在内存里"""
    files = [src] if src.is_file() else sorted(src.iterdir())
    pages: list[tuple[str, np.ndarray]] = []
    for f in files:
        if f.suffix.lower() == ".pdf":
            pages.extend(_render_pdf(f))
        elif f.suffix.lower() in IMAGE_EXTS:
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is not None:
                pages.append((f.stem, _limit(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), RENDER_MAX_SIDE)))
    return pages


def bgr(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[:, :, ::-1])


# ---------- 工具 ----------


class VramSampler:
    """后台线程按 20ms 采样整张卡的显存占用（onnxruntime 不走 torch 的显存统计，只能看 NVML）"""

    def __init__(self) -> None:
        self.ok = False
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.total_mb = pynvml.nvmlDeviceGetMemoryInfo(self._handle).total // 2**20
            self.ok = True
        except Exception as e:  # noqa: BLE001
            logger.warning("NVML 不可用，不测显存：%s", e)
        self.peak_mb = 0
        self.overall_peak_mb = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def used_mb(self) -> int:
        return self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used // 2**20 if self.ok else 0

    def start(self) -> None:
        if not self.ok:
            return
        self.peak_mb = self.used_mb()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            used = self.used_mb()
            self.peak_mb = max(self.peak_mb, used)
            self.overall_peak_mb = max(self.overall_peak_mb, used)
            time.sleep(0.02)

    def reset_peak(self) -> None:
        self.peak_mb = self.used_mb()


def _try(results: dict, key: str, fn):
    """跑一项探测，异常也记下来，不影响后面的项"""
    t0 = time.perf_counter()
    try:
        results[key] = fn()
    except Exception as e:  # noqa: BLE001
        results[key] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=6)}
        logger.error("%s 失败：%s", key, e)
    logger.info("%s 完成（%.1fs）", key, time.perf_counter() - t0)
    return results[key]


def _shape(v, depth: int = 0):
    """描述一个值的结构（类型 / 形状 / 键），不输出内容本身：结果里有识别文字"""
    if isinstance(v, np.ndarray):
        return f"ndarray{list(v.shape)} {v.dtype}"
    if isinstance(v, dict):
        if depth > 3:
            return "dict{...}"
        return {k: _shape(x, depth + 1) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        if not v:
            return f"{type(v).__name__}[0]"
        return [f"{type(v).__name__}[{len(v)}] of", _shape(v[0], depth + 1)]
    if isinstance(v, str):
        return f"str(len={len(v)})"
    return type(v).__name__


def _res_dict(res) -> dict:
    """PaddleOCR 3.x 的结果对象：.json 是 {'res': {...}}；同时兼容直接是 dict 的情况"""
    j = getattr(res, "json", None)
    if callable(j):
        j = j()
    if isinstance(j, dict):
        return j.get("res", j)
    return dict(res)


def _raw_dict(res) -> dict:
    """不经 JSON 序列化的原始字段（ndarray 保留），用于取坐标和图片"""
    try:
        return {k: res[k] for k in res.keys()}
    except Exception:  # noqa: BLE001
        return _res_dict(res)


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return round(s[min(len(s) - 1, int(round(q * (len(s) - 1))))], 1)


# ---------- 构造实例 ----------


def _signature(cls) -> list[str]:
    try:
        return [p for p in inspect.signature(cls.__init__).parameters if p != "self"]
    except (TypeError, ValueError):
        return []


OCR_KWARGS = {
    "text_detection_model_name": DET_MODEL,
    "text_recognition_model_name": REC_MODEL,
    "use_doc_orientation_classify": True,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,
    "text_det_limit_type": "max",
    "text_det_limit_side_len": 2500,
}


def engine_kwargs(engine: str | None, arena: str | None, mem_limit_mb: int, opts: list[str]) -> dict:
    """engine 参数，以及 onnxruntime CUDA EP 的选项：默认的显存池策略（kNextPowerOfTwo）和
    cuDNN 卷积算法搜索（EXHAUSTIVE + 最大 workspace）会把整张卡越占越满"""
    kw: dict = {"engine": engine} if engine else {}
    if engine == "onnxruntime" and (arena or mem_limit_mb or opts):
        provider: dict = {"device_id": 0}
        if arena:
            provider["arena_extend_strategy"] = arena
        if mem_limit_mb:
            provider["gpu_mem_limit"] = mem_limit_mb * 2**20
        for item in opts:
            k, _, v = item.partition("=")
            provider[k] = v
        kw["engine_config"] = {"provider_options": provider}
    return kw


def enable_arena_shrinkage() -> None:
    """每次 session.run 之后收缩 CUDA 显存池，把这次推理多占的显存还回去。
    onnxruntime 每个会话有自己的显存池，默认只增不减，几个模型的峰值会叠加起来；
    paddlex 的 ONNXRuntimeRunner 不支持传 RunOptions，只能包一层 session"""
    import onnxruntime as ort
    from paddlex.inference.models.runners import onnxruntime_runner as runner

    run_options = ort.RunOptions()
    run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
    create = runner.ONNXRuntimeRunner._create_session

    class _Session:
        def __init__(self, session) -> None:
            self._session = session

        def __getattr__(self, name):
            return getattr(self._session, name)

        def run(self, output_names, feeds):
            return self._session.run(output_names, feeds, run_options)

    runner.ONNXRuntimeRunner._create_session = lambda self: _Session(create(self))


def build_models(extra: dict, results: dict):
    """按方案的参数构造 LayoutDetection / PaddleOCR；不接受的参数逐个去掉重试并记录"""
    import paddleocr

    results["paddleocr_version"] = getattr(paddleocr, "__version__", "unknown")
    try:
        import paddlex

        results["paddlex_version"] = getattr(paddlex, "__version__", "unknown")
    except ImportError:
        results["paddlex_version"] = None
    try:
        import onnxruntime as ort

        results["onnxruntime"] = {"version": ort.__version__, "providers": ort.get_available_providers()}
    except ImportError:
        results["onnxruntime"] = None

    from paddleocr import LayoutDetection, PaddleOCR

    results["signature"] = {"LayoutDetection": _signature(LayoutDetection), "PaddleOCR": _signature(PaddleOCR)}

    def construct(cls, kwargs: dict, name: str):
        rejected: dict[str, str] = {}
        kwargs = dict(kwargs)
        while True:
            try:
                return cls(**kwargs), rejected
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"
                bad = next((k for k in kwargs if k in str(e)), None)
                if bad is None:
                    raise
                rejected[bad] = msg
                logger.warning("%s 不接受参数 %s：%s", name, bad, msg)
                kwargs.pop(bad)

    layout, rej = construct(LayoutDetection, {"model_name": LAYOUT_MODEL, **extra}, "LayoutDetection")
    results["layout_rejected_params"] = rej
    ocr, rej = construct(PaddleOCR, {**OCR_KWARGS, **extra}, "PaddleOCR")
    results["ocr_rejected_params"] = rej
    return layout, ocr


# ---------- 各项探测 ----------


def probe_layout(layout, pages, out: Path) -> dict:
    """第 2 项：LayoutDetection 结果结构、是否已按阅读顺序排列"""
    report: dict = {"pages": []}
    for name, img in pages:
        res = next(iter(layout.predict(bgr(img), batch_size=1)))
        if not report.get("structure"):
            report["structure"] = _shape(_res_dict(res))
            res.save_to_img(str(out / "layout"))
            res.save_to_json(str(out / "layout"))
        boxes = _res_dict(res).get("boxes", [])
        coords = [b.get("coordinate") for b in boxes]
        # 与「纯从上到下」排序比较：双栏页如果是阅读顺序，会和纯 y 排序不同，而与「先左栏后右栏」一致
        idx = list(range(len(coords)))
        by_y = sorted(idx, key=lambda i: (coords[i][1], coords[i][0]))
        mid = img.shape[1] / 2
        by_col = sorted(idx, key=lambda i: (coords[i][0] >= mid, coords[i][1]))
        orders = [b.get("order") for b in boxes]
        numbered = [o for o in orders if o is not None]
        report["pages"].append(
            {
                "page": name,
                "labels_without_order": sorted({b.get("label") for b in boxes if b.get("order") is None}),
                "labels_with_order": sorted({b.get("label") for b in boxes if b.get("order") is not None}),
                "order_monotonic_in_list": numbered == sorted(numbered),
                "size": list(img.shape[:2]),
                "boxes": len(boxes),
                "labels": sorted({b.get("label") for b in boxes}),
                "order_equals_y_sort": idx == by_y,
                "order_equals_column_sort": idx == by_col,
                "has_polygon": any(k for b in boxes for k in b if "poly" in k),
            }
        )
    return report


def probe_label_set() -> dict:
    """第 3 项：从模型目录的配置里读完整 label 集合"""
    import os

    import yaml

    home = Path(os.environ.get("PADDLE_PDX_CACHE_HOME", Path.home() / ".paddlex"))
    found = {}
    for cfg in home.rglob("*"):
        if LAYOUT_MODEL not in str(cfg) or cfg.name not in ("inference.yml", "config.json"):
            continue
        text = cfg.read_text(encoding="utf-8")
        data = json.loads(text) if cfg.suffix == ".json" else yaml.safe_load(text)
        labels = _find_key(data, "label_list")
        found[str(cfg)] = labels
    model_dirs = sorted({str(p.parent) for p in home.rglob("*") if LAYOUT_MODEL in str(p.parent) and p.is_file()})
    files = {d: sorted(p.name for p in Path(d).iterdir()) for d in model_dirs}
    return {"configs": found, "model_files": files}


def _find_key(data, key: str):
    if isinstance(data, dict):
        if key in data:
            return data[key]
        for v in data.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    elif isinstance(data, list):
        for v in data:
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def _ocr_lines(ocr, img: np.ndarray, **kw) -> tuple[dict, dict]:
    res = next(iter(ocr.predict(bgr(img), **kw)))
    return res, _raw_dict(res)


def _centers(raw: dict) -> dict[str, tuple[float, float]]:
    """识别文字 → 框中心；只用于在旋转前后的结果之间配对，不输出"""
    out = {}
    for text, box in zip(raw.get("rec_texts", []), raw.get("rec_boxes", [])):
        if len(text) >= 4 and text not in out:
            b = np.asarray(box, dtype=float).reshape(-1)
            out[text] = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    return out


def _rotate_point(x: float, y: float, w: int, h: int, k: int) -> tuple[float, float]:
    """直立图（宽 w 高 h）上的点，经 cv2.rotate 第 k 种旋转后的坐标"""
    if k == cv2.ROTATE_90_CLOCKWISE:
        return h - y, x
    if k == cv2.ROTATE_180:
        return w - x, h - y
    return y, w - x  # ROTATE_90_COUNTERCLOCKWISE


def probe_ocr(ocr, pages, out: Path) -> dict:
    """第 4 项：结果字段，以及方向校正后坐标是相对原图还是转正后的图"""
    name, img = pages[0]
    res, raw = _ocr_lines(ocr, img)
    res.save_to_img(str(out / "ocr"))
    res.save_to_json(str(out / "ocr"))
    report: dict = {"page": name, "structure": _shape(_res_dict(res)), "lines": len(raw.get("rec_texts", []))}
    base = _centers(raw)
    h, w = img.shape[:2]
    rotations = {}
    for label, k in (("90cw", cv2.ROTATE_90_CLOCKWISE), ("180", cv2.ROTATE_180), ("90ccw", cv2.ROTATE_90_COUNTERCLOCKWISE)):
        rimg = cv2.rotate(img, k)
        rres, rraw = _ocr_lines(ocr, rimg)
        rres.save_to_img(str(out / f"ocr-rot{label}"))
        pre = rraw.get("doc_preprocessor_res") or {}
        pre = pre if isinstance(pre, dict) else _raw_dict(pre)
        angle = pre.get("angle")
        out_img = pre.get("output_img")
        rc = _centers(rraw)
        common = [t for t in base if t in rc]
        # 坐标与直立图一致 → 相对转正后的图；与把直立图坐标做同样旋转后一致 → 相对原图
        d_upright = [np.hypot(rc[t][0] - base[t][0], rc[t][1] - base[t][1]) for t in common]
        d_input = [np.hypot(rc[t][0] - p[0], rc[t][1] - p[1]) for t in common for p in [_rotate_point(*base[t], w, h, k)]]
        rotations[label] = {
            "input_size_hw": list(rimg.shape[:2]),
            "detected_angle": angle,
            "preprocessor_keys": sorted(pre.keys()),
            "output_img_size_hw": list(out_img.shape[:2]) if isinstance(out_img, np.ndarray) else None,
            "matched_lines": len(common),
            "median_dist_to_upright_coords": round(float(np.median(d_upright)), 1) if common else None,
            "median_dist_to_input_coords": round(float(np.median(d_input)), 1) if common else None,
        }
    report["rotations"] = rotations
    return report


def probe_timing(layout, ocr, pages, sides: list[int], vram: VramSampler, tag: str) -> dict:
    """第 5 项：各长边档位的单页耗时（版面 + OCR）和显存峰值"""
    report = {}
    for side in sides:
        layout_ms, ocr_ms, total_ms = [], [], []
        vram.reset_peak()
        for _, img in pages:
            h, w = img.shape[:2]
            s = side / max(h, w)
            page = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
            t0 = time.perf_counter()
            next(iter(layout.predict(bgr(page), batch_size=1)))
            t1 = time.perf_counter()
            next(iter(ocr.predict(bgr(page), text_det_limit_type="max", text_det_limit_side_len=side)))
            t2 = time.perf_counter()
            layout_ms.append((t1 - t0) * 1000)
            ocr_ms.append((t2 - t1) * 1000)
            total_ms.append((t2 - t0) * 1000)
        # 第一页含预热，单独列出
        steady = total_ms[1:] or total_ms
        report[str(side)] = {
            "pages": len(pages),
            "first_page_ms": round(total_ms[0], 1),
            "layout_p50_ms": _pct(layout_ms[1:] or layout_ms, 0.5),
            "ocr_p50_ms": _pct(ocr_ms[1:] or ocr_ms, 0.5),
            "total_p50_ms": _pct(steady, 0.5),
            "total_p95_ms": _pct(steady, 0.95),
            "total_mean_ms": round(statistics.mean(steady), 1),
            "vram_peak_mb": vram.peak_mb if vram.ok else None,
        }
        logger.info("[%s] 长边 %d：%s", tag, side, report[str(side)])
    return report


def probe_lang_models(extra: dict, lang_pages: dict[str, list[tuple[str, np.ndarray]]], vram: VramSampler) -> dict:
    """第 6 项：韩 / 俄 / 泰识别模型在官方列表里的准确名字，能否用当前引擎加载，有没有 ONNX 文件"""
    report: dict = {}
    names: list[str] = []
    try:
        from paddlex.inference.utils import official_models

        registry = next(
            (v for v in vars(official_models).values() if isinstance(v, (dict, list, set)) and REC_MODEL in v), None
        )
        names = sorted(registry) if registry is not None else []
    except Exception as e:  # noqa: BLE001
        report["registry_error"] = f"{type(e).__name__}: {e}"
    report["all_rec_models"] = [n for n in names if n.endswith("_rec")]
    report["layout_models"] = [n for n in names if "Layout" in n]
    report["det_models"] = [n for n in names if n.endswith("_det")]

    from paddleocr import PaddleOCR

    for lang, prefixes in LANG_PREFIXES.items():
        candidates = [n for n in report["all_rec_models"] if n.split("_")[0] in prefixes]
        entry: dict = {"candidates": candidates, "tried": {}}
        for model in candidates[-2:]:  # 列表按名字排序，取最新的两个版本
            before = vram.used_mb()
            t0 = time.perf_counter()
            try:
                ocr = PaddleOCR(
                    text_detection_model_name=DET_MODEL,
                    text_recognition_model_name=model,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    **extra,
                )
                tried: dict = {"load_s": round(time.perf_counter() - t0, 1), "vram_delta_mb": vram.used_mb() - before}
                pages = lang_pages.get(lang, [])
                if pages:
                    _, raw = _ocr_lines(ocr, pages[0][1])
                    scores = list(raw.get("rec_scores", []))
                    tried["lines"] = len(scores)
                    tried["mean_score"] = round(float(np.mean(scores)), 3) if scores else None
                tried["onnx_files"] = _onnx_files(model)
                del ocr
            except Exception as e:  # noqa: BLE001
                tried = {"error": f"{type(e).__name__}: {e}"}
            entry["tried"][model] = tried
        report[lang] = entry
    return report


def _onnx_files(model: str) -> list[str]:
    import os

    home = Path(os.environ.get("PADDLE_PDX_CACHE_HOME", Path.home() / ".paddlex"))
    return sorted(str(p.relative_to(home)) for p in home.rglob("*.onnx") if model in str(p))


def probe_comic(vram: VramSampler) -> dict:
    """加载漫画流水线并跑一页，之后的文档耗时测试就是两者共存的显存"""
    from app.config import Settings
    from app.real_pipeline import RealPipeline

    before = vram.used_mb()
    pipeline = RealPipeline(Settings.from_env())
    sample = Path(__file__).resolve().parent.parent / "tests" / "samples" / "page.jpg"
    vram.reset_peak()
    t0 = time.perf_counter()
    pipeline.run(sample.read_bytes(), inpaint=True, debug=False)
    probe_comic.pipeline = pipeline  # 保持常驻
    return {
        "load_vram_delta_mb": vram.used_mb() - before,
        "run_ms": round((time.perf_counter() - t0) * 1000, 1),
        "vram_peak_after_comic_run_mb": vram.peak_mb if vram.ok else None,
    }


# ---------- 主流程 ----------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engine", default="onnxruntime", help="onnxruntime / paddle_static / none（不传 engine 参数）")
    parser.add_argument("--sides", default="2000,2500,3000")
    parser.add_argument("--pages", type=int, default=6, help="耗时测试用的页数（从全部页里均匀抽取）")
    parser.add_argument("--with-comic", action="store_true")
    parser.add_argument("--ort-arena", default="kSameAsRequested", help="onnxruntime CUDA 显存池扩展策略；none = 默认")
    parser.add_argument("--ort-mem-limit-mb", type=int, default=0, help="onnxruntime CUDA 显存池上限（每个会话）；0 = 不限")
    parser.add_argument("--ort-shrink", action="store_true", help="每次推理后收缩 onnxruntime 显存池")
    parser.add_argument("--ort-opt", action="append", default=[], help="其他 CUDA EP 选项 key=value，可重复")
    parser.add_argument("--skip-lang", action="store_true", help="跳过第 6 项")
    parser.add_argument("--skip-structure", action="store_true", help="跳过第 2-4 项（只测耗时、显存和语种模型）")
    parser.add_argument("--lang-pages", default="ko=06_korean,ru=07_russian", help="语种=页名前缀，用于第 6 项试识别")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    engine = None if args.engine == "none" else args.engine
    extra = engine_kwargs(engine, None if args.ort_arena == "none" else args.ort_arena, args.ort_mem_limit_mb, args.ort_opt)
    sides = [int(s) for s in args.sides.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)

    pages = load_pages(args.input)
    if not pages:
        print(f"{args.input} 下没有图片或 PDF")
        return 1
    logger.info("共 %d 页", len(pages))
    step = max(1, len(pages) // args.pages)
    sample = pages[::step][: args.pages]
    lang_pages = {}
    for item in args.lang_pages.split(","):
        lang, _, prefix = item.partition("=")
        lang_pages[lang] = [p for p in pages if p[0].startswith(prefix)]

    if args.ort_shrink:
        enable_arena_shrinkage()
    vram = VramSampler()
    results: dict = {
        "ort_shrink": args.ort_shrink,
        "engine_requested": engine,
        "engine_kwargs": extra,
        "input_pages": len(pages),
        "gpu_total_mb": vram.total_mb if vram.ok else None,
        "vram_baseline_mb": vram.used_mb() if vram.ok else None,
    }
    vram.start()

    def save() -> None:
        (args.out / "probe.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    models = _try(results, "build", lambda: build_models(extra, results))
    if isinstance(models, dict):  # 构造失败
        save()
        print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
        return 1
    layout, ocr = models
    results["build"] = "ok"
    results["vram_after_load_mb"] = vram.used_mb() if vram.ok else None

    if not args.skip_structure:
        _try(results, "layout", lambda: probe_layout(layout, pages, args.out))
        _try(results, "layout_labels", probe_label_set)
        _try(results, "ocr", lambda: probe_ocr(ocr, pages, args.out))
    save()
    _try(results, "timing_doc_only", lambda: probe_timing(layout, ocr, sample, sides, vram, "doc"))
    save()
    if args.with_comic:
        _try(results, "comic", lambda: probe_comic(vram))
        _try(results, "timing_with_comic", lambda: probe_timing(layout, ocr, sample, sides, vram, "doc+comic"))
        save()
    if not args.skip_lang:
        _try(results, "lang_models", lambda: probe_lang_models(extra, lang_pages, vram))
    results["vram_peak_overall_mb"] = vram.overall_peak_mb if vram.ok else None
    save()
    print(f"结论已写入 {args.out / 'probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
