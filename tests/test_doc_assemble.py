import cv2
import numpy as np
import pytest

from app.doc.assemble import (
    LayoutBox,
    OcrLine,
    assemble_figure,
    assemble_page,
    assign_lines,
    filter_detections,
    group_regions,
    is_list_item,
    join_lines,
    join_list,
    looks_like_list,
    order_lines,
    original_size,
    table_grid,
    to_original,
    vertical_columns,
    visual_rows,
)
from app.doc.labels import block_type, normalize_lang, rec_model_for


def ln(x1: float, y1: float, x2: float, y2: float, text: str = "x", conf: float = 0.9) -> OcrLine:
    return OcrLine(((x1, y1), (x2, y1), (x2, y2), (x1, y2)), text, conf)


# ---------- 拼接 ----------


def test_join_english_lines_with_space() -> None:
    assert join_lines(["Document translation is", "not solely a problem."]) == "Document translation is not solely a problem."


def test_join_restores_hyphenated_words() -> None:
    assert join_lines(["machine trans-", "lation works"]) == "machine translation works"
    # 连字符前是大写、或下一行以大写开头：保留连字符
    assert join_lines(["COVID-", "19 cases"]) == "COVID- 19 cases"
    assert join_lines(["state-of-the-", "Art"]) == "state-of-the- Art"


def test_join_chinese_without_spaces() -> None:
    assert join_lines(["查本月收发文件，计来文若干件，", "复函若干件。"]) == "查本月收发文件，计来文若干件，复函若干件。"
    # 下一行以全角标点开头
    assert join_lines(["俟核对后补录", "。"]) == "俟核对后补录。"


def test_join_japanese_kana_without_spaces() -> None:
    assert join_lines(["この資料は言語検証のために", "作成された。"]) == "この資料は言語検証のために作成された。"


def test_join_mixed_cjk_and_latin_uses_space_at_boundary() -> None:
    assert join_lines(["使用 PaddleOCR", "进行识别"]) == "使用 PaddleOCR 进行识别"
    assert join_lines(["模型名称为", "PP-OCRv6"]) == "模型名称为 PP-OCRv6"


def test_join_thai_without_spaces() -> None:
    assert join_lines(["ภาษาไทย", "ทดสอบ"]) == "ภาษาไทยทดสอบ"


def test_join_single_line_and_blank_lines() -> None:
    assert join_lines(["  1. Introduction  "]) == "1. Introduction"
    assert join_lines(["", "a", " ", "b"]) == "a b"
    assert join_lines([]) == ""


def test_list_items_detection() -> None:
    for t in ["• item", "- item", "· 项目", "1. first", "2) second", "(3) third", "① 一", "a) alpha", "（一）总则", "二、范围"]:
        assert is_list_item(t), t
    for t in ["Results suggest that", "1990 was a year", "e.g. something", "-5 degrees", "3.14 is pi"]:
        assert not is_list_item(t), t


def test_join_list_splits_items_and_joins_continuations() -> None:
    texts = ["• First item that wraps onto", "the next line", "• Second item", "1. trans-", "lation item"]
    assert looks_like_list(texts)
    assert join_list(texts) == "• First item that wraps onto the next line\n• Second item\n1. translation item"


def test_single_numbered_line_is_not_a_list() -> None:
    assert not looks_like_list(["1. Introduction"])
    assert not looks_like_list(["Plain text", "• bullet"])


# ---------- 排序 ----------


def test_visual_rows_group_by_y_and_sort_left_to_right() -> None:
    a, b = ln(300, 100, 500, 120, "b"), ln(10, 102, 200, 122, "a")
    c = ln(10, 140, 200, 160, "c")
    rows = visual_rows([c, a, b])
    assert [[x.text for x in r] for r in rows] == [["a", "b"], ["c"]]


def test_vertical_columns_right_to_left_top_to_bottom() -> None:
    right_top, right_bottom = ln(200, 0, 220, 100, "1"), ln(200, 110, 220, 200, "2")
    left = ln(150, 0, 170, 100, "3")
    assert [x.text for x in order_lines([left, right_bottom, right_top], vertical=True)] == ["1", "2", "3"]
    assert len(vertical_columns([left, right_top])) == 2


def test_assign_lines_picks_largest_overlap_above_half() -> None:
    boxes = [(0, 0, 100, 100), (90, 0, 300, 100)]
    lines = [ln(10, 10, 80, 20), ln(80, 10, 200, 20), ln(400, 10, 500, 20), ln(60, 10, 140, 20)]
    # 第 4 行在两个块里各一半（各 0.375 / 0.625）
    assert assign_lines(lines, boxes) == [0, 1, None, 1]


def test_group_regions_keeps_table_rows_apart() -> None:
    # 表格截图：行距约等于行高，各单元格不应按列合并
    lines = [ln(x, y, x + 80, y + 20, f"{x}-{y}") for y in (0, 40, 80) for x in (0, 200)]
    groups = group_regions(lines)
    assert len(groups) == 6
    assert [g[0].text for g in groups] == ["0-0", "200-0", "0-40", "200-40", "0-80", "200-80"]


def test_group_regions_orders_vertical_columns_right_to_left() -> None:
    cols = [ln(x, 0, x + 20, 300, str(x)) for x in (100, 200, 300)]  # 列间距大于列宽，各自成区域
    assert [g[0].text for g in group_regions(cols)] == ["300", "200", "100"]


def test_group_regions_merges_close_overlapping_lines() -> None:
    lines = [ln(0, 0, 100, 20, "a"), ln(0, 25, 100, 45, "b"), ln(0, 200, 100, 220, "c"), ln(300, 0, 400, 20, "d")]
    groups = group_regions(lines)
    assert [[x.text for x in g] for g in groups] == [["a", "b"], ["d"], ["c"]]


# ---------- 表格 ----------


def _table(rows: list[list[tuple[float, str]]], y0: float = 0, h: float = 20, gap: float = 10) -> list[OcrLine]:
    out = []
    for r, cells in enumerate(rows):
        y = y0 + r * (h + gap)
        out.extend(ln(x, y, x + 80, y + h, t) for x, t in cells)
    return out


def test_table_grid_rows_and_columns() -> None:
    lines = _table([[(100, "Model"), (300, "Score")], [(102, "A"), (301, "0.65")], [(99, "B"), (300, "0.72")]])
    grid = table_grid(lines, page_width=1000)
    assert grid is not None
    assert (grid.rows, grid.cols) == (3, 2)
    assert [(c.r, c.c, c.text) for c in grid.cells] == [
        (0, 0, "Model"), (0, 1, "Score"), (1, 0, "A"), (1, 1, "0.65"), (2, 0, "B"), (2, 1, "0.72"),
    ]


def test_table_grid_misaligned_returns_none() -> None:
    # 第二列的起点散乱（居中对齐）：每一「列」只在一行出现
    lines = _table([[(100, "a"), (300, "b")], [(100, "c"), (360, "d")], [(100, "e"), (420, "f")]])
    assert table_grid(lines, page_width=1000) is None
    assert table_grid(_table([[(100, "only"), (300, "row")]]), page_width=1000) is None


# ---------- 坐标还原 ----------


@pytest.mark.parametrize(
    ("angle", "rotate"),
    [(90, cv2.ROTATE_90_CLOCKWISE), (180, cv2.ROTATE_180), (270, cv2.ROTATE_90_COUNTERCLOCKWISE)],
)
def test_to_original_inverts_orientation(angle: int, rotate: int) -> None:
    """原图 = 直立页按 angle 顺时针转过（方向分类的含义）；直立页上的点映射回原图后，应正好落在标记像素上"""
    w, h = 40, 30
    upright = np.zeros((h, w), np.uint8)
    upright[7, 25] = 255
    original = cv2.rotate(upright, rotate)
    ys, xs = np.nonzero(original)
    x, y = to_original((25.5, 7.5), angle, (w, h))
    assert (int(x), int(y)) == (xs[0], ys[0])
    assert original_size(angle, (w, h)) == (original.shape[1], original.shape[0])


# ---------- 组装 ----------


def test_assemble_two_column_page_keeps_layout_order() -> None:
    layout = [
        LayoutBox("header", (0, 0, 1000, 40)),
        LayoutBox("paragraph_title", (50, 60, 450, 90)),
        LayoutBox("text", (50, 100, 450, 300)),
        LayoutBox("text", (550, 100, 950, 300)),
        LayoutBox("image", (550, 320, 950, 600)),
        LayoutBox("number", (480, 960, 520, 990)),
    ]
    lines = [
        ln(10, 10, 300, 30, "Journal header"),
        ln(50, 60, 300, 90, "1. Intro"),
        # 右栏和左栏的行交错给出（PaddleOCR 自己的顺序），组装后按块归属
        ln(550, 100, 950, 120, "right one"),
        ln(50, 100, 450, 120, "left one con-"),
        ln(550, 130, 950, 150, "right two"),
        ln(50, 130, 450, 150, "tinues here"),
        ln(600, 400, 700, 420, "text in image"),
        ln(490, 965, 510, 985, "7"),
    ]
    page = assemble_page(lines, layout, angle=0, upright_size=(1000, 1000))
    blocks = page["blocks"]
    assert [b["type"] for b in blocks] == ["skip", "heading", "paragraph", "paragraph", "figure", "skip"]
    assert [b["order"] for b in blocks] == list(range(6))
    assert blocks[2]["text"] == "left one continues here"
    assert blocks[3]["text"] == "right one right two"
    assert blocks[4]["text"] == "" and blocks[4]["lines"] == []
    assert blocks[0]["text"] == "Journal header"
    assert blocks[1]["bbox"] == [0.05, 0.06, 0.45, 0.09]
    assert blocks[2]["font_size"] == 0.02
    assert 0.0 < page["confidence"] <= 1.0


def test_assemble_keeps_orphan_lines_in_reading_order() -> None:
    layout = [LayoutBox("text", (0, 0, 500, 100)), LayoutBox("text", (0, 400, 500, 500))]
    lines = [ln(0, 10, 500, 30, "top"), ln(0, 410, 500, 430, "bottom"), ln(0, 200, 400, 220, "lost"), ln(0, 225, 400, 245, "line")]
    blocks = assemble_page(lines, layout, angle=0, upright_size=(500, 500))["blocks"]
    assert [b["text"] for b in blocks] == ["top", "lost line", "bottom"]
    assert blocks[1]["source_label"] == ""


def test_orphan_table_cells_keep_row_order() -> None:
    # 版面漏检的表格：单元格都没有归属，应按行从左到右排在上方正文之后、页码之前
    layout = [LayoutBox("text", (0, 0, 900, 100)), LayoutBox("number", (850, 900, 900, 950))]
    cells = [ln(x, y, x + 80, y + 20, f"{r}{c}") for r, y in enumerate((300, 340, 380)) for c, x in enumerate((0, 300, 600))]
    lines = [ln(0, 10, 900, 30, "body"), ln(855, 905, 895, 945, "02", 1.0), *reversed(cells)]
    blocks = assemble_page(lines, layout, angle=0, upright_size=(1000, 1000))["blocks"]
    assert [b["text"] for b in blocks] == ["body", "00", "01", "02", "10", "11", "12", "20", "21", "22", "02"]


def test_orphan_vertical_column_inserted_by_column_order() -> None:
    # 版面漏检了中间一列竖排：应排在右边一列之后、左边一列之前
    layout = [LayoutBox("vertical_text", (290, 0, 330, 400)), LayoutBox("vertical_text", (90, 0, 130, 400))]
    lines = [ln(300, 0, 320, 400, "右"), ln(100, 0, 120, 400, "左"), ln(200, 0, 220, 400, "中")]
    blocks = assemble_page(lines, layout, angle=0, upright_size=(400, 400))["blocks"]
    assert [b["text"] for b in blocks] == ["右", "中", "左"]


def test_orphan_heading_above_vertical_columns_goes_first() -> None:
    # 竖排页上版面漏检的横排标题：离它最近的是左边的列，但它在整串竖排列的上方，应排在最前面；
    # 页脚在整串竖排列的下方，应排在最后面
    layout = [LayoutBox("vertical_text", (900, 200, 940, 900)), LayoutBox("vertical_text", (600, 200, 640, 900)),
              LayoutBox("vertical_text", (300, 200, 340, 900))]
    lines = [ln(910, 200, 930, 900, "一"), ln(610, 200, 630, 900, "二"), ln(310, 200, 330, 900, "三"),
             ln(100, 50, 500, 90, "标题"), ln(100, 950, 500, 990, "页尾说明")]
    blocks = assemble_page(lines, layout, angle=0, upright_size=(1000, 1000))["blocks"]
    assert [b["text"] for b in blocks] == ["标题", "一", "二", "三", "页尾说明"]


def test_assemble_vertical_text_block() -> None:
    layout = [LayoutBox("vertical_text", (100, 0, 230, 300))]
    lines = [ln(110, 0, 130, 200, "左の列"), ln(200, 0, 220, 300, "右の列は")]
    [blk] = assemble_page(lines, layout, angle=0, upright_size=(300, 300))["blocks"]
    assert blk["vertical"] is True
    assert blk["text"] == "右の列は左の列"


def test_assemble_table_and_list_blocks() -> None:
    table_lines = _table([[(100, "Model"), (300, "Score")], [(100, "A"), (300, "0.65")]], y0=100)
    list_lines = [ln(100, 300, 600, 320, "• one"), ln(100, 330, 600, 350, "• two")]
    layout = [LayoutBox("table", (90, 90, 500, 160)), LayoutBox("text", (90, 290, 610, 360))]
    table, lst = assemble_page(table_lines + list_lines, layout, angle=0, upright_size=(1000, 1000))["blocks"]
    assert table["type"] == "table"
    assert table["text"] == "Model\tScore\nA\t0.65"
    assert table["table"]["rows"] == 2 and table["table"]["cols"] == 2
    assert lst["type"] == "list"
    assert lst["text"] == "• one\n• two"


def test_assemble_maps_coordinates_back_to_rotated_original() -> None:
    # 原图是直立页顺时针转 90°：直立页（宽 200、高 100）上左上角的块，在原图（宽 100、高 200）里位于右上角
    layout = [LayoutBox("text", (0, 0, 50, 20))]
    page = assemble_page([ln(0, 0, 50, 20, "hi")], layout, angle=90, upright_size=(200, 100))
    assert page["rotation"] == 90
    assert page["blocks"][0]["bbox"] == [0.8, 0.0, 1.0, 0.25]


def test_assemble_ignores_inline_formula_and_unknown_label() -> None:
    layout = [LayoutBox("inline_formula", (0, 0, 10, 10)), LayoutBox("brand_new_label", (0, 0, 100, 100))]
    blocks = assemble_page([ln(0, 0, 100, 20, "x")], layout, angle=0, upright_size=(100, 100))["blocks"]
    assert [(b["type"], b["source_label"]) for b in blocks] == [("paragraph", "brand_new_label")]


def test_assemble_figure_regions_and_ratio() -> None:
    lines = [
        ln(10, 10, 110, 30, "Revenue", 0.97),
        ln(10, 35, 110, 55, "(USD)", 0.95),
        ln(150, 150, 190, 170, "noise", 0.3),
    ]
    fig = assemble_figure(lines, (200, 200))
    assert fig["has_text"] is True
    assert [r["text"] for r in fig["regions"]] == ["Revenue (USD)"]
    assert fig["text_ratio"] == pytest.approx(2 * 100 * 20 / 40000)
    assert assemble_figure([], (200, 200)) == {"has_text": False, "text_ratio": 0.0, "regions": []}


def test_filter_detections_drops_tiny_and_low_score() -> None:
    polys = [[(0, 0), (100, 0), (100, 20), (0, 20)], [(0, 0), (100, 0), (100, 5), (0, 5)], [(0, 0), (50, 0), (50, 20), (0, 20)]]
    assert filter_detections(polys, [0.9, 0.9, 0.2], min_side=8, min_score=0.5) == [0]


# ---------- labels ----------


def test_block_type_and_language_routing() -> None:
    assert block_type("doc_title") == "heading"
    assert block_type("number") == "skip"
    assert block_type("display_formula") == "formula"
    assert normalize_lang("zh-CN") == "zh"
    assert normalize_lang(None) == "auto"
    assert rec_model_for("ko") == "korean_PP-OCRv5_mobile_rec"
    assert rec_model_for("uk") == "cyrillic_PP-OCRv5_mobile_rec"
    assert rec_model_for("fr") == "PP-OCRv6_medium_rec"


# ---------- 旋转、清理 ----------


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_to_upright_inverts_to_original(angle: int) -> None:
    from app.doc.assemble import to_upright

    upright = (200.0, 100.0)
    for p in [(0.0, 0.0), (30.0, 70.0), (199.0, 99.0)]:
        assert to_upright(to_original(p, angle, upright), angle, upright) == pytest.approx(p)


def test_order_quad_restores_top_left_start() -> None:
    from app.doc.assemble import order_quad

    rotated = ((10, 0), (10, 50), (0, 50), (0, 0))  # 起点是右上角
    assert order_quad(rotated) == ((0, 0), (10, 0), (10, 50), (0, 50))


def test_vertical_ratio() -> None:
    from app.doc.assemble import vertical_ratio

    tall = [(0, 0), (10, 0), (10, 100), (0, 100)]
    wide = [(0, 0), (100, 0), (100, 10), (0, 10)]
    assert vertical_ratio([tall, tall, tall, wide]) == 0.75
    assert vertical_ratio([]) == 0.0


def test_clean_text_merges_duplicated_punctuation() -> None:
    from app.doc.assemble import clean_text

    assert clean_text("查本月收发文件，，计来文") == "查本月收发文件，计来文"
    assert clean_text(",，均属虚构。。") == "，均属虚构。"
    assert clean_text("a, b") == "a, b"


def test_noise_orphans_are_dropped() -> None:
    layout = [LayoutBox("text", (0, 0, 500, 100))]
    lines = [
        ln(0, 10, 500, 30, "body"),
        ln(300, 300, 310, 310, "·", 0.99),  # 只有符号
        ln(100, 250, 120, 270, "集", 0.22),  # 污点被识别成单个汉字
        ln(400, 450, 440, 470, "02", 1.0),  # 页码：保留
        ln(0, 400, 100, 420, "orphan"),
    ]
    blocks = assemble_page(lines, layout, angle=0, upright_size=(500, 500))["blocks"]
    assert [b["text"] for b in blocks] == ["body", "orphan", "02"]


def test_block_lines_are_not_filtered_as_noise() -> None:
    layout = [LayoutBox("number", (480, 960, 520, 990))]
    [blk] = assemble_page([ln(490, 965, 510, 985, "7", 0.5)], layout, angle=0, upright_size=(1000, 1000))["blocks"]
    assert blk["text"] == "7"
