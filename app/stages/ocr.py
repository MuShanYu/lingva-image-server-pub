"""OCR：manga-ocr 全精度（kha-white/manga-ocr-base，transformers VisionEncoderDecoder）。

预处理 / 后处理与 manga-ocr 原版（manga_ocr/ocr.py）一致：灰度后转回 RGB，ViTImageProcessor，
generate(max_length=300)，沿用模型配置里的 beam search（num_beams=4）；文本清洗见 text_clean.py。
（贪心解码、int8 量化在 CPU 上能快约 2 倍，但会带来一两个字的识别差异，按「质量优先」不采用。）
GPU 上用 fp16 权重，no_repeat_ngram_size 换成 GPU 向量化实现（_NoRepeatNGram），识别结果与原版一致。
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    BertJapaneseTokenizer,
    GenerationMixin,
    LogitsProcessor,
    LogitsProcessorList,
    ViTImageProcessorPil,
    VisionEncoderDecoderModel,
)

from .text_clean import clean_text

BATCH_SIZE = 16
MAX_LENGTH = 300


class _MangaOcrModel(VisionEncoderDecoderModel, GenerationMixin):
    """transformers 5 起 PreTrainedModel 不再自带 generate，与原版一样显式混入 GenerationMixin。"""


class _NoRepeatNGram(LogitsProcessor):
    """与 transformers 的 NoRepeatNGramLogitsProcessor 规则相同（禁止生成已出现过的 n 元组），但全程在 GPU 上向量化计算。

    原实现每一步都把全部候选序列拷回 CPU、用 Python 循环逐条建 n 元组表；模型配置里 no_repeat_ngram_size=3，
    加上 beam search 的 4 倍候选，这是 GPU 上解码的主要耗时（每步约 9ms，模型本身只有 2 层解码器）。
    """

    def __init__(self, n: int) -> None:
        self._n = n

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        n, length = self._n, input_ids.shape[1]
        if length < n:
            return scores
        windows = input_ids.unfold(1, n, 1)  # [候选数, 窗口数, n]
        prefix = input_ids[:, length - n + 1 :]  # 最后 n-1 个 token
        hit = (windows[:, :, :-1] == prefix[:, None, :]).all(dim=-1)
        vocab = scores.shape[1]
        # 没命中的窗口指向多出来的一列，最后丢掉；所有写入的值都是 True，重复下标不影响结果
        banned = torch.where(hit, windows[:, :, -1], vocab)
        mask = torch.zeros(scores.shape[0], vocab + 1, dtype=torch.bool, device=scores.device)
        mask.scatter_(1, banned, True)
        return scores.masked_fill(mask[:, :vocab], float("-inf"))


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float  # 生成 token 的平均概率


class MangaOcr:
    def __init__(self, model_dir: Path, device: str) -> None:
        self._device = device
        # PIL 后端与原版训练/推理时的预处理一致，也不需要 torchvision
        self._processor = ViTImageProcessorPil.from_pretrained(model_dir)
        # 分词器只用于解码（id → 字符）。配置里的 MeCab 分词只在编码文本时才需要，
        # 这里改用 basic，避免额外依赖 fugashi / unidic_lite，解码结果不变
        self._tokenizer = BertJapaneseTokenizer.from_pretrained(model_dir, word_tokenizer_type="basic")
        self._model = _MangaOcrModel.from_pretrained(model_dir).to(device).eval()
        # GPU 上整体用 fp16：测试集 351 张裁图的识别结果与 fp32 完全一致（autocast 与 bf16 都会有个别字不同）；
        # generate 内部会把 logits 转回 fp32 再做 beam search 打分
        self._dtype = torch.float16 if device == "cuda" else torch.float32
        self._model.to(self._dtype)
        # 用向量化实现替换配置里的 no_repeat_ngram_size，规则不变
        self._no_repeat = self._model.generation_config.no_repeat_ngram_size or 0
        self._logits_processor = LogitsProcessorList([_NoRepeatNGram(self._no_repeat)] if self._no_repeat else [])

    @torch.no_grad()
    def __call__(self, crops: list[Image.Image]) -> list[OcrResult]:
        # 按宽高比（≈字数）排序后再分批：同一批里长短相近，生成步数不会被一条长文本拖着空转
        order = sorted(range(len(crops)), key=lambda i: max(crops[i].size) / max(1, min(crops[i].size)))
        results: list[OcrResult | None] = [None] * len(crops)
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            batch = [crops[i].convert("L").convert("RGB") for i in idx]
            pixel_values = self._processor(batch, return_tensors="pt").pixel_values.to(self._device, self._dtype)
            out = self._model.generate(
                pixel_values,
                max_length=MAX_LENGTH,
                no_repeat_ngram_size=0,
                logits_processor=self._logits_processor,
                return_dict_in_generate=True,
                output_scores=True,
            )
            seqs = out.sequences
            # 模型配置默认 beam search（num_beams=4）：按 beam_indices 取每步所选 token 的 log 概率，
            # 结束后的填充位置（beam_indices == -1）不计入。sequences_scores 带 length_penalty=2，不能直接当概率
            logprobs = self._model.compute_transition_scores(seqs, out.scores, out.beam_indices)
            valid = out.beam_indices[:, : logprobs.shape[1]] >= 0
            probs = (logprobs.exp() * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

            texts = self._tokenizer.batch_decode(seqs, skip_special_tokens=True)
            for i, t, p in zip(idx, texts, probs):
                results[i] = OcrResult(clean_text(t), float(p))
        return [r for r in results if r is not None]
