"""OCR 文本清洗与过滤（纯逻辑）。"""

import re
import unicodedata

import jaconv

# 单字块的置信度下限：低于此值且只有 1 个字符时视为误检（如网点、线条被识别成一个字）
SINGLE_CHAR_CONF_MIN = 0.8
# 任意块的置信度下限。manga-ocr 遇到非日文（英文水印等）、过小或混入邻块文字的裁图时会「编造」
# 通顺但无关的句子；测试集上这类结果置信度为 0.45~0.72，正确识别的最低为 0.80
CONF_MIN = 0.75

_DOTS = re.compile(r"[．・.]{2,}")


def clean_text(raw: str) -> str:
    """manga-ocr 原版后处理 + 统一省略号。

    原版：去掉所有空白、半角转全角（ASCII 与数字）。原版把「…」展开成「...」再转全角，
    这里把连续的点统一回全角省略号「…」（每 3 个点一个，至少一个），便于翻译。
    """
    text = "".join(raw.split())
    text = text.replace("…", "...")
    text = jaconv.h2z(text, ascii=True, digit=True)
    return _DOTS.sub(lambda m: "…" * max(1, round(len(m.group()) / 3)), text)


def _is_symbol(ch: str) -> bool:
    return unicodedata.category(ch)[0] in ("P", "S") or ch in "ー～〜"


def is_noise(text: str, confidence: float) -> bool:
    """文本为空、只有标点符号、置信度过低，或只有 1 个字且置信度偏低时，视为无需翻译的误检。"""
    if not text or confidence < CONF_MIN:
        return True
    if all(_is_symbol(ch) for ch in text):
        return True
    return len(text) == 1 and confidence < SINGLE_CHAR_CONF_MIN
