"""版面 label → 块类型，以及 lang_hint → 识别模型。label 集合来自 PP-DocLayoutV3 的 inference.yml
（notes/doc-ocr-probe.md 第 3 节）。"""

import logging
from typing import Literal

logger = logging.getLogger("lingva")

BlockType = Literal["heading", "paragraph", "list", "table", "figure", "formula", "skip"]

LABEL_TYPES: dict[str, BlockType] = {
    "doc_title": "heading",
    "paragraph_title": "heading",
    "text": "paragraph",
    "abstract": "paragraph",
    "content": "paragraph",
    "reference": "paragraph",
    "reference_content": "paragraph",
    "footnote": "paragraph",
    "aside_text": "paragraph",
    "figure_title": "paragraph",  # 模型没有 table_title / chart_title，表题、图题都归这里
    "vision_footnote": "paragraph",  # 图表下方的注释
    "algorithm": "paragraph",
    "vertical_text": "paragraph",  # 同时作为竖排的判定依据
    "table": "table",
    "image": "figure",
    "chart": "figure",
    "seal": "figure",
    "header_image": "figure",
    "footer_image": "figure",
    "display_formula": "formula",
    "formula_number": "formula",
    "header": "skip",
    "footer": "skip",
    "number": "skip",  # 页码
}
# 行内公式由覆盖它的文字行识别，不单独成块
IGNORED_LABELS = {"inline_formula"}
VERTICAL_LABEL = "vertical_text"
# 版面里没有对应块的行聚成的段落，source_label 用这个值
ORPHAN_LABEL = ""

_warned: set[str] = set()


def block_type(label: str) -> BlockType:
    """未知 label 按段落处理（模型升级后可能出现新 label），每个只警告一次"""
    t = LABEL_TYPES.get(label)
    if t is None:
        if label not in _warned:
            _warned.add(label)
            logger.warning("未知的版面 label：%s，按 paragraph 处理", label)
        return "paragraph"
    return t


DEFAULT_REC = "PP-OCRv6_medium_rec"
DET_MODEL = "PP-OCRv6_medium_det"
LAYOUT_MODEL = "PP-DocLayoutV3"
ORIENT_MODEL = "PP-LCNet_x1_0_doc_ori"

# PP-OCRv6 覆盖中、英、日和拉丁字母语言；韩文、西里尔字母、泰文用 v5 的语种模型——v6 识别这几种文字时
# 一个字符都对不上，置信度却很高，靠置信度发现不了（探针第 6 节）
_CYRILLIC = {"ru", "uk", "be", "bg", "sr", "mk", "kk", "ky", "mn", "tg", "tt", "ba", "cv", "sah", "ce", "ab", "os"}
LANG_REC: dict[str, str] = {
    "ko": "korean_PP-OCRv5_mobile_rec",
    "th": "th_PP-OCRv5_mobile_rec",
    **{lang: "cyrillic_PP-OCRv5_mobile_rec" for lang in _CYRILLIC},
}


def normalize_lang(hint: str | None) -> str:
    """'zh-CN' → 'zh'，空值 → 'auto'"""
    if not hint:
        return "auto"
    return hint.strip().lower().replace("_", "-").split("-")[0] or "auto"


def rec_model_for(lang: str) -> str:
    return LANG_REC.get(lang, DEFAULT_REC)
