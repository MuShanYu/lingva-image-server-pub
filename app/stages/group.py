"""文字行 → 块，以及阅读顺序。纯逻辑，不依赖模型。

分组规则（每一步只处理前一步没有归属的行）：
1. 行按 IoA（交集 / 行面积）≥ IOA_MIN 归入气泡检测模型给出的文字区域（text_bubble / text_free），
   同一区域内的行合并为一个块。一个气泡里有两段分开的文字时，会得到两个块。
2. 剩余的行按 IoA 归入气泡框（bubble），同一气泡内的行合并为一个块。
3. 仍剩余的行按「同方向、字号相近、间距小、另一轴有重叠」两两合并（并查集），作为 free 块。
落在同一气泡框内的多个组，如果彼此有可合并的行（同一规则），再合并为一个块——检测模型有时会把
同一段文字的每一行各标成一个文字区域；间距大的两段文字仍保持分开。
块的类型：块框落在某个气泡框内 → bubble（bubble_bbox 为该气泡框）；否则来自 text_bubble 区域的块
仍视为 bubble，bubble_bbox 退化为该文字区域；其余为 free。
兜底：高分文字区域里一条 CTD 行都没有、或者行只覆盖了区域很小一部分时（CTD 漏检），
把整个区域当作一行，单独成块。
去重：一个块的框大部分（≥ CONTAINED_IOA）落在另一个块里时合并为一块（不同气泡的块除外）；
兜底块与普通块合并时丢弃兜底生成的那条「整区域行」。
异常行：CTD 偶尔把相邻的几行文字、甚至文字框连同边上的字识别成一条粗得离谱的「行」。
粗细超过全页中位数 LINE_OUTLIER_RATIO 倍、且里面包含文字区域的行直接丢弃，交给区域兜底。
另一种异常行：一半压在文字区域上、一半压在旁边的画面线条（速度线、排线）上，而且比区域里的
文字粗得多。它归不进区域，单独成块后会读出不存在的文字、把旁边的线条擦掉，同样丢弃。
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Literal

from ..schemas import BBox

IOA_MIN = 0.5
MERGE_GAP = 1.0  # 行间距上限，单位为字号
MERGE_SIZE_RATIO = 1.6  # 两行字号之比上限
MERGE_OVERLAP_MIN = 0.3  # 另一轴重叠长度 / 较短一方长度
SAME_BAND_OVERLAP = 0.5  # 垂直重叠超过较小块高度的该比例时视为同一横带
SAME_COLUMN_OVERLAP = 0.5  # 带内水平重叠超过较窄块宽度的该比例时视为同一列
ORPHAN_REGION_SCORE_MIN = 0.7  # 没有 CTD 行的文字区域，分数不低于此值才兜底成块
CONTAINED_IOA = 0.7  # 较小块的框落在较大块内的比例达到此值时合并
CONTINUATION_OVERLAP = 0.5  # 同一行被切成两段：横排时上下重叠 / 竖排时左右重叠至少占较短一方的比例
LINE_OUTLIER_RATIO = 4.0  # 行粗细超过全页中位数的倍数
REGION_COVERAGE_MIN = 0.25  # 区域内所有行的外接框覆盖区域面积低于此比例时，按 CTD 漏检处理
STRADDLE_IOA_MIN = 0.3  # 行与文字区域的 IoA 在此值与 IOA_MIN 之间，视为「一半压在区域上」
STRADDLE_THICK_RATIO = 1.5  # 行的粗细超过区域短边的这个倍数，说明它不是区域里的一行文字


@dataclass(frozen=True)
class LineIn:
    bbox: BBox
    vertical: bool


@dataclass(frozen=True)
class TextRegion:
    bbox: BBox
    free: bool  # True = text_free（气泡外文字），False = text_bubble
    score: float = 1.0


@dataclass(frozen=True)
class GroupedBlock:
    kind: Literal["bubble", "free"]
    bbox: BBox
    bubble_bbox: BBox | None
    vertical: bool
    lines: list[LineIn]  # 块内按阅读顺序排列
    font_size: int
    from_region_only: bool = False  # True = CTD 漏检、由文字区域兜底生成，没有对应的分割 mask
    region_bbox: BBox | None = None  # 所属的高分文字区域（分数 ≥ ORPHAN_REGION_SCORE_MIN），没有时为 None


def area(b: BBox) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def intersection(a: BBox, b: BBox) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def ioa(inner: BBox, outer: BBox) -> float:
    a = area(inner)
    return intersection(inner, outer) / a if a > 0 else 0.0


def union_bbox(boxes: Sequence[BBox]) -> BBox:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def best_container(box: BBox, containers: Sequence[BBox], min_ioa: float = IOA_MIN) -> int | None:
    """IoA 最大且 ≥ min_ioa 的容器下标；IoA 相同时取面积较小的（更贴近的）容器。"""
    best: int | None = None
    best_key: tuple[float, int] | None = None
    for i, c in enumerate(containers):
        v = ioa(box, c)
        if v < min_ioa:
            continue
        key = (v, -area(c))
        if best_key is None or key > best_key:
            best, best_key = i, key
    return best


def thickness(line: LineIn) -> int:
    """行的「粗细」≈ 字号：竖排取宽度，横排取高度。"""
    x1, y1, x2, y2 = line.bbox
    return (x2 - x1) if line.vertical else (y2 - y1)


def _span_overlap(a1: int, a2: int, b1: int, b2: int) -> int:
    return max(0, min(a2, b2) - max(a1, b1))


def _span_gap(a1: int, a2: int, b1: int, b2: int) -> int:
    return max(0, max(a1, b1) - min(a2, b2))


def can_merge(a: LineIn, b: LineIn) -> bool:
    if a.vertical != b.vertical:
        return False
    ta, tb = thickness(a), thickness(b)
    if min(ta, tb) <= 0 or max(ta, tb) / min(ta, tb) > MERGE_SIZE_RATIO:
        return False
    size = max(ta, tb)
    (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = a.bbox, b.bbox
    x_gap, y_gap = _span_gap(ax1, ax2, bx1, bx2), _span_gap(ay1, ay2, by1, by2)
    x_ov = _span_overlap(ax1, ax2, bx1, bx2) / min(ax2 - ax1, bx2 - bx1)
    y_ov = _span_overlap(ay1, ay2, by1, by2) / min(ay2 - ay1, by2 - by1)
    if a.vertical:
        # 相邻的两列（左右相邻、上下有重叠），或同一列被切成了上下两段
        stacked = x_gap <= MERGE_GAP * size and y_ov >= MERGE_OVERLAP_MIN
        continued = y_gap <= MERGE_GAP * size and x_ov >= CONTINUATION_OVERLAP
    else:
        # 相邻的两行（上下相邻、左右有重叠），或同一行被切成了左右两段
        stacked = y_gap <= MERGE_GAP * size and x_ov >= MERGE_OVERLAP_MIN
        continued = x_gap <= MERGE_GAP * size and y_ov >= CONTINUATION_OVERLAP
    return stacked or continued


def _connected(items: Sequence[int], linked: Callable[[int, int], bool]) -> list[list[int]]:
    """按 linked 关系求连通分量（并查集），保持 items 中的原有顺序。"""
    parent = {i: i for i in items}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for pos, i in enumerate(items):
        for j in items[pos + 1 :]:
            if linked(i, j):
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in items:
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def merge_nearby(lines: Sequence[LineIn], indices: Sequence[int]) -> list[list[int]]:
    return _connected(indices, lambda i, j: can_merge(lines[i], lines[j]))


def _center(b: BBox) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _same_band(a: BBox, b: BBox) -> bool:
    return _span_overlap(a[1], a[3], b[1], b[3]) > SAME_BAND_OVERLAP * min(a[3] - a[1], b[3] - b[1])


def _same_column(a: BBox, b: BBox) -> bool:
    return _span_overlap(a[0], a[2], b[0], b[2]) > SAME_COLUMN_OVERLAP * min(a[2] - a[0], b[2] - b[0])


def reading_order(boxes: Sequence[BBox]) -> list[int]:
    """日漫阅读顺序，返回按阅读顺序排列的下标。

    横带（垂直方向重叠的块，可传递）从上到下；带内再按水平重叠分列，列从右到左、列内从上到下。
    分列是为了处理一个很高的块把旁边上下堆叠的多个小块拉进同一横带的情况。
    """
    idx = list(range(len(boxes)))
    bands = _connected(idx, lambda i, j: _same_band(boxes[i], boxes[j]))
    bands.sort(key=lambda band: min(boxes[j][1] for j in band))

    result: list[int] = []
    for band in bands:
        columns = _connected(band, lambda i, j: _same_column(boxes[i], boxes[j]))
        columns.sort(key=lambda col: -max(_center(boxes[j])[0] for j in col))
        for col in columns:
            result += sorted(col, key=lambda k: _center(boxes[k])[1])
    return result


def order_lines(lines: Sequence[LineIn], vertical: bool) -> list[LineIn]:
    """块内行的阅读顺序。竖排：列从右到左，同一列被切成几段时从上到下；
    横排：行从上到下，同一行被切成几段时从左到右。"""
    idx = list(range(len(lines)))
    boxes = [ln.bbox for ln in lines]
    if vertical:
        cols = _connected(idx, lambda i, j: _same_column(boxes[i], boxes[j]))
        cols.sort(key=lambda c: -max(_center(boxes[k])[0] for k in c))
        return [lines[k] for c in cols for k in sorted(c, key=lambda k: boxes[k][1])]
    rows = _connected(idx, lambda i, j: _same_band(boxes[i], boxes[j]))
    rows.sort(key=lambda r: min(boxes[k][1] for k in r))
    return [lines[k] for r in rows for k in sorted(r, key=lambda k: boxes[k][0])]


def drop_outlier_lines(lines: Sequence[LineIn], regions: Sequence[TextRegion]) -> list[LineIn]:
    if len(lines) < 3:
        return list(lines)
    med = median(thickness(ln) for ln in lines)
    return [
        ln
        for ln in lines
        if not (
            thickness(ln) > LINE_OUTLIER_RATIO * med
            and any(ioa(r.bbox, ln.bbox) >= IOA_MIN for r in regions)
        )
    ]


def drop_straddling_lines(lines: Sequence[LineIn], regions: Sequence[TextRegion]) -> list[LineIn]:
    def straddles(ln: LineIn) -> bool:
        for r in regions:
            x1, y1, x2, y2 = r.bbox
            if (
                STRADDLE_IOA_MIN <= ioa(ln.bbox, r.bbox) < IOA_MIN
                and thickness(ln) > STRADDLE_THICK_RATIO * min(x2 - x1, y2 - y1)
            ):
                return True
        return False

    return [ln for ln in lines if not straddles(ln)]


@dataclass
class _Group:
    members: list[int]
    region: TextRegion | None
    bubble: int | None
    from_region_only: bool = False


def _merge_within_bubbles(lines: Sequence[LineIn], groups: list[_Group]) -> list[_Group]:
    def linked(a: int, b: int) -> bool:
        ga, gb = groups[a], groups[b]
        return (
            ga.bubble is not None
            and ga.bubble == gb.bubble
            and any(can_merge(lines[i], lines[j]) for i in ga.members for j in gb.members)
        )

    merged: list[_Group] = []
    for comp in _connected(list(range(len(groups))), linked):
        first = groups[comp[0]]
        members = [m for g in comp for m in groups[g].members]
        merged.append(_Group(members, first.region, first.bubble))
    return merged


def _merge_contained(lines: Sequence[LineIn], groups: list[_Group]) -> list[_Group]:
    boxes = [union_bbox([lines[i].bbox for i in g.members]) for g in groups]

    def linked(a: int, b: int) -> bool:
        ga, gb = groups[a], groups[b]
        if ga.bubble is not None and gb.bubble is not None and ga.bubble != gb.bubble:
            return False
        small, big = sorted((a, b), key=lambda k: area(boxes[k]))
        return ioa(boxes[small], boxes[big]) >= CONTAINED_IOA

    merged: list[_Group] = []
    for comp in _connected(list(range(len(groups))), linked):
        real = [k for k in comp if not groups[k].from_region_only]
        if not real:
            merged.append(groups[max(comp, key=lambda k: area(boxes[k]))])
            continue
        largest = groups[max(real, key=lambda k: area(boxes[k]))]
        members = [m for k in real for m in groups[k].members]
        merged.append(_Group(members, largest.region, largest.bubble))
    return merged


def _make_block(
    lines: Sequence[LineIn], group: _Group, bubbles: Sequence[BBox], page_font: int | None
) -> GroupedBlock:
    members, region = group.members, group.region
    member_lines = [lines[i] for i in members]
    vertical_votes = sum(1 for ln in member_lines if ln.vertical)
    vertical = vertical_votes * 2 >= len(member_lines)  # 平票按竖排（日漫默认）
    bbox = union_bbox([ln.bbox for ln in member_lines])

    b = group.bubble if group.bubble is not None else best_container(bbox, bubbles)
    if b is not None:
        kind: Literal["bubble", "free"] = "bubble"
        bubble_bbox: BBox | None = bubbles[b]
    elif region is not None and not region.free:
        kind, bubble_bbox = "bubble", region.bbox
    else:
        kind, bubble_bbox = "free", None

    ordered = order_lines(member_lines, vertical)
    if group.from_region_only:
        # 区域框代表整块文字而不是一行，粗细不等于字号：按短边估计（单行 / 单列区域时准确），
        # 但不超过全页文字行粗细的中位数（多行区域的短边是好几行的高度，会把膨胀量放得过大）
        font_size = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
        if page_font:
            font_size = min(font_size, page_font)
    else:
        font_size = round(median(thickness(ln) for ln in member_lines))
    trusted = region is not None and region.score >= ORPHAN_REGION_SCORE_MIN
    return GroupedBlock(
        kind,
        bbox,
        bubble_bbox,
        vertical,
        ordered,
        max(1, font_size),
        group.from_region_only,
        region.bbox if trusted else None,
    )


def group_lines(
    lines: Sequence[LineIn],
    regions: Sequence[TextRegion],
    bubbles: Sequence[BBox],
    order: Callable[[Sequence[BBox]], list[int]] = reading_order,
) -> list[GroupedBlock]:
    """把文字行分组成块，并按 order 排好阅读顺序后返回。"""
    lines = drop_straddling_lines(drop_outlier_lines(lines, regions), regions)
    remaining = list(range(len(lines)))

    by_region: dict[int, list[int]] = {}
    region_boxes = [r.bbox for r in regions]
    for i in list(remaining):
        r = best_container(lines[i].bbox, region_boxes)
        if r is not None:
            by_region.setdefault(r, []).append(i)
            remaining.remove(i)

    # 行只覆盖了高分区域的一小部分：CTD 漏检了区域里的大部分文字，改用整个区域兜底（下面的循环处理）
    for r, members in list(by_region.items()):
        covered = intersection(union_bbox([lines[i].bbox for i in members]), regions[r].bbox)
        if regions[r].score >= ORPHAN_REGION_SCORE_MIN and covered < REGION_COVERAGE_MIN * area(regions[r].bbox):
            del by_region[r]

    by_bubble: dict[int, list[int]] = {}
    for i in list(remaining):
        b = best_container(lines[i].bbox, bubbles)
        if b is not None:
            by_bubble.setdefault(b, []).append(i)
            remaining.remove(i)

    groups = [
        _Group(m, regions[r], best_container(union_bbox([lines[i].bbox for i in m]), bubbles))
        for r, m in by_region.items()
    ]
    groups += [_Group(m, None, b) for b, m in by_bubble.items()]
    groups = _merge_within_bubbles(lines, groups)
    groups += [_Group(m, None, None) for m in merge_nearby(lines, remaining)]

    all_lines = list(lines)
    for r, region in enumerate(regions):
        if r in by_region or region.score < ORPHAN_REGION_SCORE_MIN:
            continue
        x1, y1, x2, y2 = region.bbox
        all_lines.append(LineIn(region.bbox, vertical=(y2 - y1) > (x2 - x1)))
        groups.append(_Group([len(all_lines) - 1], region, best_container(region.bbox, bubbles), True))

    groups = _merge_contained(all_lines, groups)
    page_font = round(median(thickness(ln) for ln in lines)) if lines else None
    blocks = [_make_block(all_lines, g, bubbles, page_font) for g in groups]
    return [blocks[i] for i in order([blk.bbox for blk in blocks])]
