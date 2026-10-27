from typing import Literal

from pydantic import BaseModel

# [x1, y1, x2, y2]，原图（EXIF 转正后）像素坐标，x2/y2 不含
BBox = tuple[int, int, int, int]


class Line(BaseModel):
    bbox: BBox


class Block(BaseModel):
    id: int
    order: int
    kind: Literal["bubble", "free"]
    bbox: BBox
    bubble_bbox: BBox | None
    vertical: bool
    text: str
    lines: list[Line]
    font_size: int
    fg_color: str
    bg_color: str


class Capabilities(BaseModel):
    inpaint: bool
    upscale: bool


class DocPipelineInfo(BaseModel):
    enabled: bool
    # layout / det / orientation：模型名；rec：已加载的识别模型（语种模型按需加载）
    models: dict[str, str | list[str]] = {}
    erase: bool = False  # /v1/doc/page 支持 erase；旧版本服务没有这个字段


class HealthResponse(BaseModel):
    api_version: int
    stub: bool
    device: str
    model_version: dict[str, str]
    capabilities: Capabilities
    # 文档识别管线；旧版本服务没有这个字段，客户端按未启用处理
    doc_pipeline: DocPipelineInfo = DocPipelineInfo(enabled=False)


# ---- 文档识别（/v1/doc/page、/v1/doc/figure）：坐标都是相对原图（EXIF 转正后）的归一化坐标 [0, 1] ----

NBox = tuple[float, float, float, float]  # [x1, y1, x2, y2]


class DocLine(BaseModel):
    text: str
    bbox: NBox
    poly: list[tuple[float, float]]
    conf: float


class DocTableCell(BaseModel):
    r: int
    c: int
    text: str
    bbox: NBox


class DocTable(BaseModel):
    rows: int
    cols: int
    cells: list[DocTableCell]


class DocBlock(BaseModel):
    id: int
    type: Literal["heading", "paragraph", "list", "table", "figure", "formula", "skip"]
    source_label: str  # 版面模型的原始 label；空串表示版面里没有对应块的行聚成的段落
    order: int  # 阅读顺序，从 0 开始连续
    bbox: NBox
    text: str  # 拼好的整段文字；figure / formula 为空串；表格每行一行、单元格以制表符分隔
    font_size: float  # 块内行粗细中位数 / 转正后的页面高度
    vertical: bool
    lines: list[DocLine]
    table: DocTable | None  # 仅 type=table 且网格对齐时有值


class DocPageResponse(BaseModel):
    mode: Literal["doc_page"] = "doc_page"
    rotation: Literal[0, 90, 180, 270]  # 原图内容被顺时针转过的角度，坐标已逆变换回原图
    lang: str  # 实际使用的识别语种；语种模型不可用、降级到默认模型时为 auto
    rec_model: str
    confidence: float  # 按字符数加权的平均识别置信度
    blocks: list[DocBlock]
    erased_image: str | None = None  # erase=true 且有文字行时为擦掉所有行后的整页图：base64 WebP，尺寸和方向同上传的图
    timing_ms: dict[str, int]


class DocRegion(BaseModel):
    id: int
    bbox: NBox
    text: str
    vertical: bool
    lines: list[DocLine]


class DocFigureResponse(BaseModel):
    mode: Literal["doc_figure"] = "doc_figure"
    has_text: bool  # false 时不扣额度
    text_ratio: float  # 文字框面积 / 图片面积
    lang: str
    rec_model: str
    regions: list[DocRegion]
    erased_image: str | None  # erase=true 且有文字时为擦字后的图片：base64 WebP（同 clean_image）
    timing_ms: dict[str, int]


class AnalyzeResponse(BaseModel):
    width: int
    height: int
    model_version: dict[str, str]
    blocks: list[Block]
    clean_image: str | None
    debug_mask: str | None
    timings_ms: dict[str, int]


class ErrorResponse(BaseModel):
    code: str  # 稳定的机器可读错误码，取值见 app/errors.py 的 Code
    error: str  # 英文说明，仅供调试，客户端不要依赖
