from app.stages.group import (
    LineIn,
    TextRegion,
    best_container,
    can_merge,
    group_lines,
    ioa,
    reading_order,
)


def v(x1: int, y1: int, x2: int, y2: int) -> LineIn:
    return LineIn((x1, y1, x2, y2), vertical=True)


def h(x1: int, y1: int, x2: int, y2: int) -> LineIn:
    return LineIn((x1, y1, x2, y2), vertical=False)


def test_ioa_is_relative_to_inner_area() -> None:
    assert ioa((0, 0, 10, 10), (0, 0, 100, 100)) == 1.0
    assert ioa((0, 0, 10, 10), (5, 0, 100, 100)) == 0.5
    assert ioa((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_best_container_requires_min_ioa_and_prefers_tighter() -> None:
    outer, inner = (0, 0, 200, 200), (0, 0, 50, 50)
    assert best_container((10, 10, 20, 20), [outer, inner]) == 1
    assert best_container((40, 40, 60, 60), [(0, 0, 45, 45)]) is None


def test_lines_in_text_region_become_one_bubble_block() -> None:
    lines = [v(100, 20, 120, 200), v(70, 20, 90, 150)]
    regions = [TextRegion((60, 10, 130, 210), free=False)]
    bubbles = [(40, 0, 150, 230)]
    [blk] = group_lines(lines, regions, bubbles)
    assert blk.kind == "bubble"
    assert blk.bubble_bbox == (40, 0, 150, 230)
    assert blk.bbox == (70, 20, 120, 200)
    assert blk.vertical is True
    assert blk.lines == [lines[0], lines[1]]  # 竖排从右往左


def test_two_text_regions_in_one_bubble_stay_separate() -> None:
    lines = [v(300, 20, 320, 100), v(100, 20, 120, 100)]
    regions = [TextRegion((290, 10, 330, 110), False), TextRegion((90, 10, 130, 110), False)]
    bubbles = [(80, 0, 340, 120)]
    blocks = group_lines(lines, regions, bubbles)
    assert len(blocks) == 2
    assert all(b.bubble_bbox == (80, 0, 340, 120) for b in blocks)
    assert blocks[0].lines == [lines[0]]  # 右边的先读


def test_adjacent_regions_in_same_bubble_are_merged() -> None:
    # 检测模型把同一段横排文字的两行各标成一个区域
    lines = [h(20, 20, 200, 40), h(40, 46, 220, 66)]
    regions = [TextRegion((15, 15, 205, 42), False), TextRegion((35, 44, 225, 68), False)]
    [blk] = group_lines(lines, regions, [(0, 0, 240, 80)])
    assert blk.lines == [lines[0], lines[1]]
    assert blk.bbox == (20, 20, 220, 66)


def test_adjacent_regions_in_different_bubbles_stay_separate() -> None:
    lines = [h(20, 20, 200, 40), h(40, 46, 220, 66)]
    regions = [TextRegion((15, 15, 205, 42), False), TextRegion((35, 44, 225, 68), False)]
    blocks = group_lines(lines, regions, [(0, 0, 240, 43), (0, 43, 240, 80)])
    assert len(blocks) == 2


def test_high_score_region_without_lines_becomes_block() -> None:
    regions = [TextRegion((10, 10, 40, 200), False, score=0.9), TextRegion((300, 10, 330, 60), True, score=0.4)]
    [blk] = group_lines([], regions, [(0, 0, 50, 220)])
    assert blk.from_region_only is True
    assert blk.kind == "bubble"
    assert blk.bubble_bbox == (0, 0, 50, 220)
    assert blk.bbox == (10, 10, 40, 200)
    assert blk.vertical is True
    assert blk.font_size == 30


def test_region_with_lines_is_not_duplicated() -> None:
    lines = [v(10, 10, 30, 100)]
    blocks = group_lines(lines, [TextRegion((5, 5, 35, 105), False, score=0.95)], [])
    assert len(blocks) == 1
    assert blocks[0].from_region_only is False


def test_block_contained_in_another_is_merged() -> None:
    # 两个重叠的文字区域各分到了行，其中一个块几乎完全落在另一个块里
    lines = [h(10, 10, 200, 30), h(10, 40, 200, 60), h(20, 42, 120, 58)]
    regions = [TextRegion((5, 5, 205, 65), True, 0.9), TextRegion((18, 40, 125, 60), True, 0.8)]
    [blk] = group_lines(lines, regions, [])
    assert blk.bbox == (10, 10, 200, 60)
    assert len(blk.lines) == 3


def test_orphan_region_overlapping_block_is_dropped() -> None:
    lines = [h(10, 10, 200, 30)]
    regions = [TextRegion((5, 5, 205, 35), True, 0.9), TextRegion((50, 12, 120, 28), True, 0.95)]
    # 第二个区域没分到行（行归属 IoA 更大的第一个区域），但它落在已有块里，不再单独成块
    blocks = group_lines(lines, regions, [])
    assert len(blocks) == 1
    assert blocks[0].from_region_only is False
    assert blocks[0].lines == [lines[0]]


def test_contained_blocks_in_different_bubbles_stay_separate() -> None:
    lines = [v(100, 10, 120, 200), v(105, 20, 115, 60)]
    regions = [TextRegion((95, 5, 125, 205), False), TextRegion((104, 19, 116, 61), False)]
    bubbles = [(80, 0, 140, 220), (100, 15, 120, 65)]
    assert len(group_lines(lines, regions, bubbles)) == 2


def test_text_bubble_region_without_bubble_falls_back_to_region() -> None:
    lines = [v(10, 10, 30, 100)]
    [blk] = group_lines(lines, [TextRegion((5, 5, 35, 105), False)], [])
    assert blk.kind == "bubble"
    assert blk.bubble_bbox == (5, 5, 35, 105)


def test_text_free_region_is_free() -> None:
    lines = [h(10, 10, 200, 30), h(10, 40, 180, 60)]
    [blk] = group_lines(lines, [TextRegion((0, 0, 210, 70), True)], [])
    assert blk.kind == "free"
    assert blk.bubble_bbox is None
    assert blk.vertical is False
    assert blk.lines == [lines[0], lines[1]]  # 横排从上往下


def test_lines_without_region_fall_back_to_bubble() -> None:
    lines = [v(100, 20, 120, 200), v(70, 20, 90, 150)]
    [blk] = group_lines(lines, [], [(40, 0, 150, 230)])
    assert blk.kind == "bubble"
    assert blk.bubble_bbox == (40, 0, 150, 230)


def test_region_below_min_ioa_is_ignored() -> None:
    lines = [v(0, 0, 20, 100)]
    [blk] = group_lines(lines, [TextRegion((0, 60, 20, 200), False, score=0.5)], [])
    assert blk.kind == "free"


def test_free_lines_merge_by_proximity() -> None:
    lines = [v(100, 0, 120, 100), v(75, 0, 95, 80), v(400, 0, 420, 100)]
    blocks = group_lines(lines, [], [])
    groups = sorted(sorted(lines.index(ln) for ln in b.lines) for b in blocks)
    assert groups == [[0, 1], [2]]


def test_merge_rejects_mixed_direction_and_size() -> None:
    assert not can_merge(v(0, 0, 20, 100), h(25, 0, 125, 20))
    assert not can_merge(v(0, 0, 20, 100), v(25, 0, 65, 100))  # 字号 20 vs 40
    assert not can_merge(v(0, 0, 20, 100), v(25, 150, 45, 250))  # 上下没有重叠
    assert can_merge(h(0, 0, 100, 20), h(0, 30, 80, 50))


def test_split_horizontal_line_is_merged_and_read_left_to_right() -> None:
    # 「え！？そーなの！？」被 CTD 切成左右两段
    lines = [h(160, 10, 370, 40), h(60, 12, 157, 38)]
    [blk] = group_lines(lines, [], [])
    assert blk.lines == [lines[1], lines[0]]


def test_split_vertical_column_is_merged_and_read_top_down() -> None:
    lines = [v(10, 120, 30, 200), v(10, 10, 30, 110)]
    [blk] = group_lines(lines, [], [])
    assert blk.lines == [lines[1], lines[0]]


def test_multi_row_horizontal_block_order() -> None:
    lines = [h(10, 40, 100, 60), h(110, 10, 200, 30), h(10, 10, 100, 30)]
    [blk] = group_lines(lines, [TextRegion((0, 0, 210, 70), True)], [])
    assert blk.lines == [lines[2], lines[1], lines[0]]


def test_outlier_thick_line_is_dropped_in_favor_of_region() -> None:
    normal = [h(500, 10, 700, 35), h(500, 50, 700, 75), h(500, 90, 700, 115)]
    giant = h(10, 30, 350, 225)  # 把一个评论框整个框成了一行
    regions = [TextRegion((20, 70, 315, 185), True, 0.9)]
    blocks = group_lines(normal + [giant], regions, [])
    region_blocks = [b for b in blocks if b.from_region_only]
    assert len(region_blocks) == 1 and region_blocks[0].bbox == (20, 70, 315, 185)
    assert all(giant not in b.lines for b in blocks)


def test_sparse_lines_in_region_fall_back_to_region() -> None:
    lines = [h(75, 458, 205, 485)]  # 评论框里 CTD 只检出最后一行
    [blk] = group_lines(lines, [TextRegion((45, 365, 445, 490), True, 0.95)], [])
    assert blk.from_region_only is True
    assert blk.bbox == (45, 365, 445, 490)


def test_block_direction_is_majority_and_font_size_is_median() -> None:
    lines = [v(100, 0, 120, 100), v(70, 0, 92, 100), h(60, 110, 130, 128)]
    [blk] = group_lines(lines, [TextRegion((50, 0, 140, 140), False)], [])
    assert blk.vertical is True
    assert blk.font_size == 20  # 20、22、18 的中位数


def test_reading_order_bands_top_down_right_to_left() -> None:
    boxes = [
        (0, 0, 50, 100),  # 左上
        (200, 10, 250, 90),  # 右上
        (100, 300, 150, 400),  # 下方
        (100, 20, 150, 110),  # 中上
    ]
    assert reading_order(boxes) == [1, 3, 0, 2]


def test_reading_order_small_overlap_starts_new_band() -> None:
    boxes = [(0, 0, 50, 100), (200, 90, 250, 190)]  # 只重叠 10px
    assert reading_order(boxes) == [0, 1]


def test_reading_order_stacked_blocks_beside_tall_block() -> None:
    boxes = [
        (400, 0, 450, 400),  # 右侧很高的竖排块，把左边的块都拉进同一横带
        (0, 300, 300, 340),  # 左侧第三个
        (0, 20, 300, 60),  # 左侧第一个
        (0, 160, 300, 200),  # 左侧第二个
    ]
    assert reading_order(boxes) == [0, 2, 3, 1]


def test_group_lines_accepts_custom_order() -> None:
    lines = [v(0, 0, 20, 100), v(300, 0, 320, 100)]
    blocks = group_lines(lines, [], [], order=lambda boxes: list(range(len(boxes))))
    assert [b.bbox[0] for b in blocks] == [0, 300]


def test_thick_line_straddling_a_region_is_dropped() -> None:
    # CTD 把文字列的上半截连同旁边的速度线框成了一条很粗的「行」：一半在区域里、比区域宽得多
    blob = v(545, 17, 722, 263)
    regions = [TextRegion((626, 10, 710, 489), True, 0.73)]
    [blk] = group_lines([blob], regions, [])
    assert blk.from_region_only is True
    assert blk.bbox == (626, 10, 710, 489)


def test_normal_line_partly_overlapping_a_region_is_kept() -> None:
    neighbour = v(590, 20, 650, 260)  # 粗细和区域里的文字差不多：是旁边的另一列字
    regions = [TextRegion((626, 10, 710, 489), True, 0.73)]
    blocks = group_lines([neighbour], regions, [])
    assert any(neighbour in b.lines for b in blocks)


def test_block_records_high_score_region_only() -> None:
    lines = [v(100, 20, 120, 200)]
    [high] = group_lines(lines, [TextRegion((90, 10, 130, 210), True, 0.9)], [])
    [low] = group_lines(lines, [TextRegion((90, 10, 130, 210), True, 0.4)], [])
    [none] = group_lines(lines, [], [])
    assert high.region_bbox == (90, 10, 130, 210)
    assert low.region_bbox is None and none.region_bbox is None
