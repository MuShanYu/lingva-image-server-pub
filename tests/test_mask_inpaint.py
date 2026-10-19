import numpy as np

from app.stages.color import estimate_colors, is_colored_text, is_plain_background
from app.stages.group import GroupedBlock, LineIn
from app.stages.inpaint import fill_flat_regions, flat_fill_color, plan_regions
from app.stages.mask import (
    block_dilation,
    block_text_masks,
    erase_mask,
    extend_to_regions,
    otsu_text_mask,
    residue_mask,
)


def white_page(h: int = 200, w: int = 200) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)


def block(bbox, kind="bubble", bubble_bbox=None, font_size=20, from_region_only=False) -> GroupedBlock:
    return GroupedBlock(kind, bbox, bubble_bbox, True, [LineIn(bbox, True)], font_size, from_region_only)


def _residue_scene():
    original = white_page()
    original[80:120, 80:120] = 0  # 原文
    original[95:105, 122:125] = 0  # mask 边上没擦到的一小截笔画（如「○」的残边）
    original[74:75, 0:200] = 0  # 穿过检查范围（mask 外扩 6px）的长线（线稿）
    clean = original.copy()
    clean[80:120, 80:120] = 255  # 原文已擦掉
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[78:122, 78:122] = 255
    seg = np.zeros((200, 200), dtype=np.uint8)
    seg[78:125, 78:125] = 255
    return original, clean, mask, seg


def test_residue_mask_finds_isolated_leftover_stroke() -> None:
    original, clean, mask, seg = _residue_scene()
    res = residue_mask(original, clean, mask, seg)
    assert res[100, 123] == 255


def test_residue_mask_ignores_line_art_and_non_text() -> None:
    original, clean, mask, seg = _residue_scene()
    seg[70:80, :] = 255  # 即使分割图误标了线，伸出检查范围的长线也不算残留
    res = residue_mask(original, clean, mask, seg)
    assert res[74, 100] == 0
    seg2 = np.zeros_like(seg)
    seg2[80:118, 80:118] = 255  # 残边不在分割图（膨胀 3px 后）里 → 不是文字，不动
    assert residue_mask(original, clean, mask, seg2)[100, 123] == 0


def test_block_dilation_scales_with_font_and_is_capped() -> None:
    assert block_dilation(20, 30) == 4
    assert block_dilation(5, 30) == 3
    assert block_dilation(400, 30) == 30


def test_text_mask_only_inside_line_zone() -> None:
    img = white_page()
    seg = np.zeros((200, 200), dtype=np.uint8)
    seg[50:60, 50:60] = 255  # 行内的字
    seg[150:160, 150:160] = 255  # 行外被误判成文字的线条
    [m] = block_text_masks(img, seg, [block((45, 45, 65, 65))])
    assert m[55, 55]
    assert not m[155, 155]


def test_region_only_block_uses_otsu() -> None:
    img = white_page()
    img[60:70, 60:64] = 0
    seg = np.zeros((200, 200), dtype=np.uint8)
    [m] = block_text_masks(img, seg, [block((50, 50, 80, 80), from_region_only=True)])
    assert m[65, 62] and not m[55, 55]


def test_otsu_picks_minority_class_as_text() -> None:
    gray = np.full((20, 20), 20, dtype=np.uint8)  # 黑底
    gray[5:8, 5:15] = 240  # 白字
    m = otsu_text_mask(gray)
    assert m[6, 10] and not m[0, 0]


def test_erase_mask_dilates_clips_to_bubble_and_keeps_line_art() -> None:
    img = white_page()
    img[:, 120] = 0  # 气泡框外、离文字很近的竖线（线稿）
    text = np.zeros((200, 200), dtype=bool)
    text[90:110, 100:110] = True
    blk = block((100, 90, 110, 110), bubble_bbox=(0, 0, 118, 200), font_size=40)
    mask = erase_mask(img, [text], [blk], max_dilation=30)
    assert mask[100, 105] == 255  # 文字本身
    assert mask[100, 114] == 255  # 膨胀出的白边
    assert mask[100, 119] == 0  # 超出气泡框
    assert mask[100, 120] == 0


def test_erase_mask_protects_line_art_away_from_strokes() -> None:
    img = white_page()
    img[100, 115] = 140  # 离笔画 6px、在膨胀范围内的浅灰细线
    img[100, 111] = 140  # 离笔画 2px：视为文字的抗锯齿边缘
    text = np.zeros((200, 200), dtype=bool)
    text[90:110, 100:110] = True
    blk = block((100, 90, 110, 110), kind="free", font_size=40)
    mask = erase_mask(img, [text], [blk], max_dilation=30)
    assert mask[100, 115] == 0
    assert mask[100, 114] == 255
    assert mask[100, 111] == 255


def test_plan_regions_merges_nearby_components() -> None:
    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    mask[10:20, 40:50] = 255  # 相距 20px，合并
    mask[200:210, 200:210] = 255
    assert sorted(plan_regions(mask, gap=32)) == [(10, 10, 50, 20), (200, 200, 210, 210)]


def test_flat_fill_color_detects_uniform_background() -> None:
    crop = white_page(40, 40)
    m = np.zeros((40, 40), dtype=bool)
    m[15:25, 15:25] = True
    assert tuple(flat_fill_color(crop, m)) == (255, 255, 255)
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
    assert flat_fill_color(noisy, m) is None


def test_fill_flat_regions_fills_only_uniform_components() -> None:
    img = white_page()
    img[:, 100:] = np.random.default_rng(1).integers(0, 255, (200, 100, 3), dtype=np.uint8)
    img[20:30, 20:30] = 0  # 白底上的字
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[18:32, 18:32] = 255
    mask[150:160, 150:160] = 255  # 花纹背景
    out = img.copy()
    rest, filled = fill_flat_regions(out, mask)
    assert filled == 1
    assert (out[20:30, 20:30] == 255).all()
    assert rest[25, 25] == 0 and rest[155, 155] == 255


def test_estimate_colors() -> None:
    img = white_page()
    img[50:60, 50:60] = (200, 0, 0)
    text = np.zeros((200, 200), dtype=bool)
    text[50:60, 50:60] = True
    assert estimate_colors(img, text, (40, 40, 70, 70)) == ("#c80000", "#ffffff")
    assert estimate_colors(img, np.zeros((200, 200), dtype=bool), (40, 40, 70, 70))[0] == "#000000"


def test_estimate_colors_ignores_antialiased_edges() -> None:
    img = white_page()
    text = np.zeros((200, 200), dtype=bool)
    text[50:60, 50:60] = True
    img[50:60, 50:60] = 200  # mask 覆盖的浅灰抗锯齿边缘
    img[52:58, 52:58] = 0  # 笔画本体（36 px，少于边缘的 64 px）
    fg, _ = estimate_colors(img, text, (40, 40, 70, 70))
    assert fg == "#000000"


RED = (230, 20, 30)


def test_is_colored_text_looks_at_stroke_pixels_only() -> None:
    img = white_page()
    img[90:110, 100:110] = RED
    thick = np.zeros((200, 200), dtype=bool)
    thick[86:114, 96:114] = True  # 分割 mask 比笔画粗一圈，含白底
    assert is_colored_text(img, thick)
    img[90:110, 100:110] = 0
    assert not is_colored_text(img, thick)  # 黑字
    assert not is_colored_text(img, np.zeros((200, 200), dtype=bool))


def test_is_plain_background() -> None:
    img = white_page()
    text = np.zeros((200, 200), dtype=bool)
    text[90:110, 100:110] = True
    img[90:110, 100:110] = RED
    assert is_plain_background(img, text)  # 白底红字
    img[:, :100] = (250, 200, 80)  # 彩页：文字以外一半是彩色
    assert not is_plain_background(img, text)


def test_erase_mask_erases_colored_text_but_still_protects_black_line_art() -> None:
    img = white_page()
    img[90:110, 100:110] = RED
    img[100, 116] = RED  # 离分割笔画 6px 的红色像素（分割图没盖全的笔画）：是文字，要擦
    img[95, 116] = 0  # 同样距离的黑色像素：线稿，保留
    text = np.zeros((200, 200), dtype=bool)
    text[90:110, 100:110] = True
    blk = block((100, 90, 110, 110), kind="free", font_size=40)
    mask = erase_mask(img, [text], [blk], max_dilation=30)
    assert mask[100, 116] == 255
    assert mask[95, 116] == 0


def test_erase_mask_black_text_keeps_colored_art_protected() -> None:
    img = white_page()
    img[90:110, 100:110] = 0
    img[100, 116] = (150, 20, 30)  # 黑字旁边的深红色画面：仍按线稿保护
    text = np.zeros((200, 200), dtype=bool)
    text[90:110, 100:110] = True
    blk = block((100, 90, 110, 110), kind="free", font_size=40)
    assert erase_mask(img, [text], [blk], max_dilation=30)[100, 116] == 0


def test_region_only_block_adds_segmentation_to_otsu() -> None:
    img = white_page()
    img[60:70, 60:64] = 0
    seg = np.zeros((200, 200), dtype=np.uint8)
    seg[58:72, 70:76] = 255  # Otsu 没分出来、但分割图里有的笔画
    [m] = block_text_masks(img, seg, [block((50, 50, 80, 80), from_region_only=True)])
    assert m[65, 62] and m[65, 72]


def test_colored_text_mask_drops_uncolored_art() -> None:
    img = white_page()
    img[60:90, 56:62] = RED  # 红字
    img[50:100, 70:85] = 0  # 压在文字旁边的黑发
    seg = np.zeros((200, 200), dtype=np.uint8)
    seg[58:92, 54:64] = 255
    seg[55:60, 72:80] = 255  # 分割图把一小撮头发也标成了文字
    for region_only in (True, False):  # 兜底块（Otsu 会把整片黑发算进来）和普通块都一样
        [m] = block_text_masks(img, seg, [block((50, 50, 90, 100), from_region_only=region_only)])
        assert m[75, 58]
        assert not m[75, 78] and not m[57, 76]


def test_black_text_mask_keeps_segmentation_as_is() -> None:
    img = white_page()
    img[60:90, 56:62] = 0
    seg = np.zeros((200, 200), dtype=np.uint8)
    seg[58:92, 54:64] = 255
    [m] = block_text_masks(img, seg, [block((50, 50, 90, 100))])
    assert m[58:92, 54:64].all()


def test_extend_to_regions_when_segmented_text_falls_outside_block() -> None:
    seg = np.zeros((300, 200), dtype=np.uint8)
    seg[20:280, 90:110] = 255  # 一整列文字
    region = (80, 10, 120, 290)
    short = GroupedBlock("free", (88, 18, 112, 150), None, True, [LineIn((88, 18, 112, 150), True)], 24, False, region)
    full = GroupedBlock("free", (88, 18, 112, 282), None, True, [LineIn((88, 18, 112, 282), True)], 24, False, region)
    no_region = GroupedBlock("free", (88, 18, 112, 150), None, True, [LineIn((88, 18, 112, 150), True)], 24)
    a, b, c = extend_to_regions(seg, [short, full, no_region])
    assert a.bbox == (80, 10, 120, 290)  # CTD 行框只盖住上半截 → 扩到区域
    assert a.lines == short.lines
    assert b.bbox == full.bbox  # 行框已经盖住全部文字：区域框只是松一圈，不扩
    assert c.bbox == no_region.bbox
