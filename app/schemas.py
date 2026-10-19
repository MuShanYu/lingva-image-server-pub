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


class HealthResponse(BaseModel):
    api_version: int
    stub: bool
    device: str
    model_version: dict[str, str]
    capabilities: Capabilities


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
