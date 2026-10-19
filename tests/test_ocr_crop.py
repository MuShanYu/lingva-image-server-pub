import numpy as np
from PIL import Image

from app.stages.group import GroupedBlock, LineIn
from app.stages.ocr_crop import (
    CHROMA_PARAMS,
    CropParams,
    chroma_image,
    masked_crop,
    ocr_crops,
    split_long_line,
    split_points,
    text_rows,
)


def page(w: int = 400, h: int = 200) -> Image.Image:
    return Image.new("RGB", (w, h), "white")


def test_text_rows_from_projection() -> None:
    text = np.zeros((100, 50), dtype=bool)
    text[5:21, :] = True  # 第 1 行
    text[10, :] = False  # 行内的一条细空白（如「一」上下）不应断行
    text[30:46, :] = True  # 第 2 行
    text[55:88, :] = True  # 两行粘在一起（高 33 ≈ 2 × 16）
    text[95:97, :] = True  # 噪点
    assert text_rows(text, 16) == [(5, 21), (30, 46), (55, 71), (71, 88)]


def test_ocr_crops_uses_mask_rows_for_horizontal_block() -> None:
    img = page()
    img.paste((0, 0, 0), (12, 12, 100, 28))
    img.paste((0, 0, 0), (12, 42, 340, 58))
    seg = np.zeros((200, 400), dtype=np.uint8)
    seg[10:60, 10:345] = 255  # CTD 分割图笔画很粗，行间没有空隙
    # CTD 行框也很乱（一个框跨两行），切行按笔画投影来
    horiz = block([LineIn((10, 10, 350, 60), False)], vertical=False)
    crops = ocr_crops(img, seg, horiz, [horiz])
    # 按分割图切出两行：短行 88x16 → 96x24（含边距）1 段；长行 328x16 → 336x24 → 3 段
    assert len(crops) == 1 + 3
    assert {c.height for c in crops} == {24}


def test_split_points_prefer_ink_gaps() -> None:
    ink = np.full(100, 10.0)
    ink[47] = 0  # 等分点 50 附近的字间空隙
    assert split_points(ink, 2, 10) == [47]


def test_split_long_line_piece_count() -> None:
    crop = Image.new("RGB", (300, 20), "white")
    seg = np.zeros((20, 300), dtype=np.uint8)
    assert len(split_long_line(crop, seg)) == 3  # 宽高比 15 → 3 段
    short = Image.new("RGB", (90, 20), "white")
    assert len(split_long_line(short, np.zeros((20, 90), dtype=np.uint8))) == 1


def test_masked_crop_whitens_foreign_lines_but_keeps_own() -> None:
    img = page()
    img.paste((0, 0, 0), (10, 10, 60, 30))  # 自己的行
    img.paste((0, 0, 0), (62, 10, 90, 30))  # 贴着的邻块文字
    crop = masked_crop(img, (10, 10, 60, 30), [(10, 10, 60, 30)], [(62, 10, 90, 30)])
    arr = np.asarray(crop)
    # 裁图带 4px 边距：裁图坐标 = 原图坐标 - 6
    assert (arr[10, 10] == 0).all()  # 自己的行仍是黑的
    assert (arr[10, 57] == 255).all()  # 邻块（原图 x=63）被涂白


def block(lines, vertical, region_only=False) -> GroupedBlock:
    from app.stages.group import union_bbox

    return GroupedBlock("free", union_bbox([ln.bbox for ln in lines]), None, vertical, lines, 20, region_only)


def test_crop_params_change_margin_and_split() -> None:
    img = page()
    seg = np.zeros((200, 400), dtype=np.uint8)
    vert = block([LineIn((370, 10, 390, 190), True)], vertical=True)
    assert ocr_crops(img, seg, vert, [vert], CropParams(margin=10)).pop().size == (40, 200)
    horiz = block([LineIn((10, 40, 350, 60), False)], vertical=False)
    # 分割图为空时退回 CTD 行框：340x20 + 边距 1 → 342x22，宽高比 15.5；≤4 → 4 段，≤6 → 3 段
    assert len(ocr_crops(img, seg, horiz, [horiz], CropParams(margin=1, max_aspect=4.0))) == 4
    assert len(ocr_crops(img, seg, horiz, [horiz], CropParams(margin=1, max_aspect=6.0))) == 3


def test_ocr_crops_per_line_for_horizontal_and_whole_for_vertical() -> None:
    img = page()
    seg = np.zeros((200, 400), dtype=np.uint8)
    horiz = block([LineIn((10, 10, 110, 30), False), LineIn((10, 40, 350, 60), False)], vertical=False)
    vert = block([LineIn((370, 10, 390, 190), True), LineIn((345, 10, 365, 150), True)], vertical=True)
    region = block([LineIn((10, 100, 300, 180), False)], vertical=False, region_only=True)
    blocks = [horiz, vert, region]
    # 分割图为空时退回用 CTD 行框：短行 108x28 → 1 段，长行 348x28（含边距）→ 3 段
    assert len(ocr_crops(img, seg, horiz, blocks)) == 1 + 3
    assert len(ocr_crops(img, seg, vert, blocks)) == 1
    assert len(ocr_crops(img, seg, region, blocks)) == 1


def test_chroma_image_keeps_colored_text_and_drops_black_art() -> None:
    img = page()
    img.paste((230, 20, 30), (10, 10, 30, 30))  # 红字
    img.paste((0, 0, 0), (40, 10, 60, 30))  # 黑发
    arr = np.asarray(chroma_image(img))
    assert arr[20, 20].max() < 30  # 红字变黑
    assert (arr[20, 50] == 255).all()  # 黑色画面变白
    assert (arr[100, 100] == 255).all()


def test_chroma_params_return_chroma_crops() -> None:
    img = page()
    img.paste((0, 0, 0), (372, 12, 388, 188))
    seg = np.zeros((200, 400), dtype=np.uint8)
    vert = block([LineIn((370, 10, 390, 190), True)], vertical=True)
    [plain] = ocr_crops(img, seg, vert, [vert])
    [chroma] = ocr_crops(img, seg, vert, [vert], CHROMA_PARAMS)
    assert chroma.size == plain.size
    assert np.asarray(plain).min() == 0 and np.asarray(chroma).min() == 255  # 黑字在色度图里消失
