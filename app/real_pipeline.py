import logging

import numpy as np
from PIL import Image

from .config import Settings
from .device import flush_denormal, resolve_device, resolve_inpaint_precision
from .pipeline import StageTimer
from .schemas import AnalyzeResponse, Block, Capabilities, Line
from .stages.color import estimate_colors, is_colored_text, is_plain_background
from .stages.decode import decode_image
from .stages.detect_bubble import BUBBLE, TEXT_FREE, BubbleDetector
from .stages.detect_text import TextDetector
from .stages.encode import encode_png_b64, encode_webp_b64
from .stages.group import GroupedBlock, LineIn, TextRegion, group_lines
from .stages.inpaint import LamaInpainter, inpaint_image
from .stages.mask import TEXT_THRESH, block_text_masks, erase_mask, extend_to_regions, residue_mask
from .stages.ocr import MangaOcr, OcrResult
from .stages.ocr_crop import CHROMA_PARAMS, DEFAULT_PARAMS, RETRY_PARAMS, CropParams, ocr_crops
from .stages.text_clean import is_noise

logger = logging.getLogger("lingva")


OCR_RETRY_CONF = 0.9  # 置信度低于此值的块换几组裁图参数（彩色字再加色度图）重新识别，取置信度最高的结果


def _ocr_jobs(
    ocr: MangaOcr,
    img: Image.Image,
    seg: np.ndarray,
    blocks: list[GroupedBlock],
    jobs: list[tuple[int, CropParams]],
) -> list[OcrResult]:
    """一次批量识别多个（块, 裁图参数）组合。每块可能对应多张裁图（横排逐行、长行切段），
    识别后按块拼接；块的置信度取各段按字数加权的平均值。"""
    crops_per_job = [ocr_crops(img, seg, blocks[i], blocks, params) for i, params in jobs]
    flat = ocr([c for crops in crops_per_job for c in crops])
    results: list[OcrResult] = []
    pos = 0
    for crops in crops_per_job:
        parts = flat[pos : pos + len(crops)]
        pos += len(crops)
        text = "".join(p.text for p in parts)
        weight = sum(max(1, len(p.text)) for p in parts)
        conf = sum(p.confidence * max(1, len(p.text)) for p in parts) / weight if parts else 0.0
        results.append(OcrResult(text, conf))
    return results


def _chroma_readable(img: np.ndarray, seg: np.ndarray, block: GroupedBlock) -> bool:
    """黑白画面上的彩色字：色度图里只剩文字，值得用它再识别一次。"""
    x1, y1, x2, y2 = block.bbox
    region, text = img[y1:y2, x1:x2], seg[y1:y2, x1:x2] > TEXT_THRESH
    return is_colored_text(region, text) and is_plain_background(region, text)


def _recognize(
    ocr: MangaOcr, img: Image.Image, seg: np.ndarray, blocks: list[GroupedBlock]
) -> list[OcrResult]:
    if not blocks:
        return []
    best = _ocr_jobs(ocr, img, seg, blocks, [(i, DEFAULT_PARAMS) for i in range(len(blocks))])
    retry = [i for i, r in enumerate(best) if r.confidence < OCR_RETRY_CONF]
    if not retry:
        return best
    jobs = [(i, params) for params in RETRY_PARAMS for i in retry]
    rgb = np.asarray(img)
    jobs += [(i, CHROMA_PARAMS) for i in retry if _chroma_readable(rgb, seg, blocks[i])]
    improved = 0
    for (i, _), alt in zip(jobs, _ocr_jobs(ocr, img, seg, blocks, jobs)):
        if alt.confidence > best[i].confidence:
            best[i] = alt
            improved += 1
    logger.info("OCR 重试 %d 个低置信度块，%d 次得到更高置信度的结果", len(retry), improved)
    return best


class RealPipeline:
    stub = False

    def __init__(self, settings: Settings) -> None:
        self.device = resolve_device(settings.device)
        dtype, self.inpaint_precision = resolve_inpaint_precision(self.device, settings.inpaint_precision)
        self._mask_dilation = settings.mask_dilation
        self._webp_quality = settings.webp_quality

        models = settings.model_dir
        self._text = TextDetector(models / "detection" / "comictextdetector.pt", self.device, settings.detect_size)
        self._bubble = BubbleDetector(models / "bubble" / "detector.onnx", self.device, settings.bubble_score_min)
        self._ocr = MangaOcr(models / "manga-ocr-base", self.device)
        self._lama = LamaInpainter(
            models / "inpainting" / "lama_large_512px.ckpt", self.device, dtype, settings.inpaint_size
        )
        self.model_version = {
            "text_detector": "ctd-1f90fa60",
            "bubble_detector": "ogkalu-ctbd-16e8a622",
            "ocr": "manga-ocr-base-aa6573bd",
            "inpainter": "lama_large_512px-11d30fbb",
        }
        self.capabilities = Capabilities(inpaint=True, upscale=False)
        logger.info(
            "设备 %s，擦字精度 %s，气泡检测 %s", self.device, self.inpaint_precision, self._bubble.provider
        )

    def run(self, data: bytes, *, inpaint: bool, debug: bool) -> AnalyzeResponse:
        flush_denormal()
        timer = StageTimer()
        with timer.stage("decode"):
            pil_img = decode_image(data)
            img = np.asarray(pil_img)
        h, w = img.shape[:2]

        with timer.stage("detect_text"):
            det = self._text(img)
        with timer.stage("detect_bubble"):
            regions = self._bubble(img)
        with timer.stage("group"):
            grouped = group_lines(
                [LineIn(ln.bbox, ln.vertical) for ln in det.lines],
                [TextRegion(r.bbox, r.label == TEXT_FREE, r.score) for r in regions if r.label != BUBBLE],
                [r.bbox for r in regions if r.label == BUBBLE],
            )
            grouped = extend_to_regions(det.mask, grouped)

        with timer.stage("ocr"):
            ocr_results = _recognize(self._ocr, pil_img, det.mask, grouped)
        kept = [(g, r) for g, r in zip(grouped, ocr_results) if not is_noise(r.text, r.confidence)]
        dropped = len(grouped) - len(kept)
        if dropped:
            # 只记数量：识别出的文字是用户在看的内容，不写进日志
            logger.info("过滤 %d 个噪声块", dropped)
        kept_blocks = [g for g, _ in kept]

        need_mask = inpaint or debug
        mask: np.ndarray | None = None
        text_masks: list[np.ndarray] = []
        with timer.stage("mask"):
            text_masks = block_text_masks(img, det.mask, kept_blocks)
            if need_mask:
                mask = (
                    erase_mask(img, text_masks, kept_blocks, self._mask_dilation)
                    if kept_blocks
                    else np.zeros((h, w), dtype=np.uint8)
                )

        with timer.stage("color"):
            colors = [estimate_colors(img, tm, g.bbox) for tm, g in zip(text_masks, kept_blocks)]

        clean = img
        if inpaint and mask is not None:
            with timer.stage("inpaint"):
                clean, stats = inpaint_image(img, mask, self._lama)
            logger.info("擦字区域 %d 个：LaMa %d，纯色填充 %d", stats.regions, stats.lama, stats.flat)
            with timer.stage("residue"):
                residue = residue_mask(img, clean, mask, det.mask)
                if residue.any():
                    clean, stats = inpaint_image(clean, residue, self._lama)
                    mask |= residue
                    logger.info("补擦残留 %d 像素，区域 %d 个", int((residue > 0).sum()), stats.regions)

        blocks = [
            Block(
                id=i,
                order=i,
                kind=g.kind,
                bbox=g.bbox,
                bubble_bbox=g.bubble_bbox,
                vertical=g.vertical,
                text=r.text,
                lines=[Line(bbox=ln.bbox) for ln in g.lines],
                font_size=g.font_size,
                fg_color=fg,
                bg_color=bg,
            )
            for i, ((g, r), (fg, bg)) in enumerate(zip(kept, colors))
        ]

        clean_image: str | None = None
        debug_mask: str | None = None
        with timer.stage("encode"):
            if inpaint:
                clean_image = encode_webp_b64(Image.fromarray(clean), self._webp_quality)
            if debug and mask is not None:
                debug_mask = encode_png_b64(Image.fromarray(mask))

        return AnalyzeResponse(
            width=w,
            height=h,
            model_version=dict(self.model_version),
            blocks=blocks,
            clean_image=clean_image,
            debug_mask=debug_mask,
            timings_ms=timer.timings,
        )
