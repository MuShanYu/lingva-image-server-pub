"""文档识别结果的组装：OCR 行 + 版面块 → 带类型、带阅读顺序的块。纯逻辑，不依赖模型。

坐标约定：输入的行和版面块都在「转正后的图」上（像素）；输出全部逆旋转回原图，再归一化到 [0, 1]。

doc_page（design/文档识别管线-实现说明.md 第 4 节）：
1. 每个 OCR 行按「交集 / 行面积」归给比例最大且大于 ASSIGN_MIN 的版面块。
2. figure / formula 块里的行丢弃（图片里的文字由用户手动走 doc_figure）。
3. 不属于任何块的行聚成段落（同 doc_figure 的区域聚合），按中心点插入到最近的版面块前后。不能丢行
   （噪点除外，见下）。
4. 块内排序：竖排（label 为 vertical_text，或多数行高宽比 > VERTICAL_ASPECT）按列从右到左、列内从上到下；
   横排按视觉行从上到下、行内从左到右。
5. 拼接：衔接处两边都是不用空格分词的文字（CJK、假名、泰文）时直接拼接，否则加空格；英文断词还原。
   段落块的多数行以项目符号 / 编号开头时改判为 list，各项之间用换行分隔。
6. 表格粗网格：行按 y 聚类成表格行，x 起点一维聚类成列；对不齐时不出网格，text 按行换行拼接。
识别文字先做清理：合并识别模型在行首 / 句读处重复输出的标点（「，，」「,，」）。
没有归属的行里只有标点符号的、或只有一个字符且置信度低的（复印件上的污点被识别成「·」「集」）直接丢弃。
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import median

from .labels import IGNORED_LABELS, ORPHAN_LABEL, VERTICAL_LABEL, BlockType, block_type

Point = tuple[float, float]
BBox = tuple[float, float, float, float]

ASSIGN_MIN = 0.5  # 行与版面块的交集 / 行面积，超过才归属
VERTICAL_ASPECT = 2.0  # 行的高 / 宽超过此值视为竖排行
SAME_ROW_RATIO = 0.5  # 两行中心的垂直距离小于行高中位数的这个倍数时视为同一视觉行
# 聚合区域：两行间距小于行粗细中位数的这个倍数（且另一轴有重叠）时合并。方案定的是 1.0，实测
# （200 DPI 渲染、PP-OCRv6 检测框）段落内行距为 0.52~0.61、段落之间 0.7~1.1、表格行之间 0.94~1.06，
# 1.0 会把表格截图按列合并成一个区域
GROUP_GAP_RATIO = 0.65
TABLE_COL_GAP = 0.01  # 表格列聚类：相邻 x 起点间距小于页面宽度的这个比例视为同一列
TABLE_COL_FILL_MIN = 0.5  # 每一列至少要在这个比例的表格行里有内容，否则视为网格对不齐
FIGURE_CONF_MIN = 0.5  # doc_figure 丢弃识别置信度低于此值的行
# 没有归属的行只有一个字符（字母、数字、汉字）且置信度低于此值时视为噪点丢弃：复印件上的污点会被
# 识别成单个汉字或字母，置信度 0.15~0.83；真实的孤立短行（页码、表格单元格）置信度接近 1 或不止一个字符
ORPHAN_SINGLE_CONF_MIN = 0.9
COORD_DIGITS = 4


# ---------- 数据 ----------


@dataclass(frozen=True)
class OcrLine:
    poly: tuple[Point, ...]  # 检测框四个点（转正后的图，像素）
    text: str
    conf: float

    @property
    def bbox(self) -> BBox:
        xs = [p[0] for p in self.poly]
        ys = [p[1] for p in self.poly]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def thickness(self) -> float:
        """行的粗细（多边形的短边）：横排行是行高，竖排行是列宽"""
        if len(self.poly) == 4:
            p = self.poly
            a = (_dist(p[0], p[1]) + _dist(p[2], p[3])) / 2
            b = (_dist(p[1], p[2]) + _dist(p[3], p[0])) / 2
            return min(a, b)
        x1, y1, x2, y2 = self.bbox
        return min(x2 - x1, y2 - y1)

    @property
    def vertical(self) -> bool:
        x1, y1, x2, y2 = self.bbox
        return (y2 - y1) > VERTICAL_ASPECT * max(x2 - x1, 1e-6)


@dataclass(frozen=True)
class LayoutBox:
    label: str
    bbox: BBox  # 转正后的图，像素
    score: float = 1.0


@dataclass
class _Block:
    label: str
    type: BlockType
    bbox: BBox
    lines: list[OcrLine] = field(default_factory=list)


def _dist(a: Point, b: Point) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a: BBox, b: BBox) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _union(boxes: Sequence[BBox]) -> BBox:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def _center(b: BBox) -> Point:
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def poly_area(poly: Sequence[Point]) -> float:
    """多边形面积（鞋带公式）"""
    n = len(poly)
    s = sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1] for i in range(n))
    return abs(s) / 2


# ---------- 行归属 ----------


def assign_lines(lines: Sequence[OcrLine], boxes: Sequence[BBox]) -> list[int | None]:
    """每行归属的块下标：交集 / 行面积最大且大于 ASSIGN_MIN 的块，没有则为 None"""
    owners: list[int | None] = []
    for line in lines:
        lb = line.bbox
        area = max(_area(lb), 1e-6)
        best, best_ratio = None, ASSIGN_MIN
        for i, b in enumerate(boxes):
            ratio = _inter(lb, b) / area
            if ratio > best_ratio:
                best, best_ratio = i, ratio
        owners.append(best)
    return owners


# ---------- 排序 ----------


def _cluster_1d(values: Sequence[float], threshold: float) -> list[list[int]]:
    """按值排序后，相邻差小于 threshold 的连成一组；返回各组的下标（组按值从小到大）"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    groups: list[list[int]] = []
    for i in order:
        if groups and values[i] - values[groups[-1][-1]] < threshold:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def visual_rows(lines: Sequence[OcrLine]) -> list[list[OcrLine]]:
    """横排：按 y 中心聚成视觉行（从上到下），行内从左到右"""
    if not lines:
        return []
    h = median(ln.thickness for ln in lines)
    cy = [_center(ln.bbox)[1] for ln in lines]
    # 与行首比较，而不是与上一行比较：避免倾斜的长段落被串成一行
    order = sorted(range(len(lines)), key=lambda i: cy[i])
    rows: list[list[int]] = []
    for i in order:
        if rows and cy[i] - cy[rows[-1][0]] < SAME_ROW_RATIO * h:
            rows[-1].append(i)
        else:
            rows.append([i])
    return [sorted((lines[i] for i in r), key=lambda ln: ln.bbox[0]) for r in rows]


def vertical_columns(lines: Sequence[OcrLine]) -> list[list[OcrLine]]:
    """竖排：按 x 中心聚成列（从右到左），列内从上到下"""
    if not lines:
        return []
    w = median(ln.thickness for ln in lines)
    cx = [_center(ln.bbox)[0] for ln in lines]
    cols = _cluster_1d(cx, SAME_ROW_RATIO * w)
    return [sorted((lines[i] for i in c), key=lambda ln: ln.bbox[1]) for c in reversed(cols)]


def is_vertical_block(label: str, lines: Sequence[OcrLine]) -> bool:
    if label == VERTICAL_LABEL:
        return True
    return bool(lines) and sum(ln.vertical for ln in lines) * 2 > len(lines)


def order_lines(lines: Sequence[OcrLine], vertical: bool) -> list[OcrLine]:
    groups = vertical_columns(lines) if vertical else visual_rows(lines)
    return [ln for g in groups for ln in g]


# ---------- 拼接 ----------


def _no_space_script(ch: str) -> bool:
    """不用空格分词的文字：CJK 汉字、假名、CJK 标点、全角符号、泰文"""
    o = ord(ch)
    return (
        0x2E80 <= o <= 0x9FFF  # CJK 部首、标点、假名、汉字
        or 0xF900 <= o <= 0xFAFF  # 兼容汉字
        or 0xFF00 <= o <= 0xFFEF  # 全角 / 半角形式
        or 0x0E00 <= o <= 0x0E7F  # 泰文
        or 0x20000 <= o <= 0x3FFFF  # 扩展汉字
    )


_DUP_PUNCT = re.compile(r"([，。、；：！？])\1+")
_MIXED_COMMA = re.compile(r",，|，,")


def clean_text(text: str) -> str:
    """合并重复的全角标点：识别模型偶尔把行首或句读处的一个逗号输出成「，，」或「,，」"""
    return _DUP_PUNCT.sub(r"\1", _MIXED_COMMA.sub("，", text.strip()))


def is_symbol_only(text: str) -> bool:
    """只有标点、符号（没有字母、数字、汉字等）"""
    return not any(ch.isalnum() for ch in text)


def is_noise_orphan(line: "OcrLine") -> bool:
    """没有归属的行是否是噪点：只有标点符号，或只有一个字符且置信度低"""
    chars = sum(ch.isalnum() for ch in line.text)
    return chars == 0 or (chars == 1 and line.conf < ORPHAN_SINGLE_CONF_MIN)


def join_lines(texts: Sequence[str]) -> str:
    """把排好序的行拼成一段：两边都是 CJK / 泰文时直接拼接；英文断词（小写-↵小写）去掉连字符；其余加空格"""
    out = ""
    for t in texts:
        t = t.strip()
        if not t:
            continue
        if not out:
            out = t
        elif len(out) >= 2 and out[-1] == "-" and out[-2].islower() and t[0].islower():
            out = out[:-1] + t
        elif _no_space_script(out[-1]) and _no_space_script(t[0]):
            out += t
        else:
            out += " " + t
    return out


# 项目符号或编号：•、-、·、1.、(1)、①、a)、一、、（一）……符号后面要有空格的只限容易和正文混淆的
_BULLET = re.compile(
    r"^\s*(?:"
    r"[•●○◦▪▫■□◆◇►▶➢✓✔·∙‧]"
    r"|[-–—*+]\s"
    r"|\(?\d{1,3}[.)．、]\s?(?!\d)"
    r"|[(（]\d{1,3}[)）]"
    r"|[①-⑳❶-❿]"  # ①-⑳、❶-❿
    r"|[(（]?[a-zA-Z][)）]\s?"
    r"|[a-zA-Z]\.\s"
    r"|[(（][一二三四五六七八九十]{1,3}[)）]"
    r"|[一二三四五六七八九十]{1,3}、"
    r")"
)


def is_list_item(text: str) -> bool:
    return bool(_BULLET.match(text))


def looks_like_list(texts: Sequence[str]) -> bool:
    """第一行是列表项，且至少两行以项目符号 / 编号开头"""
    texts = [t for t in texts if t.strip()]
    return len(texts) >= 2 and is_list_item(texts[0]) and sum(map(is_list_item, texts)) >= 2


def join_list(texts: Sequence[str]) -> str:
    """列表项各占一行；没有符号的续行按 join_lines 的规则拼到当前项"""
    items: list[list[str]] = []
    for t in texts:
        if not t.strip():
            continue
        if not items or is_list_item(t):
            items.append([t])
        else:
            items[-1].append(t)
    return "\n".join(join_lines(item) for item in items)


# ---------- 表格 ----------


@dataclass(frozen=True)
class TableCell:
    r: int
    c: int
    text: str
    bbox: BBox  # 转正后的图，像素


@dataclass(frozen=True)
class TableGrid:
    rows: int
    cols: int
    cells: tuple[TableCell, ...]


def table_rows_text(rows: Sequence[Sequence[OcrLine]]) -> str:
    """表格的纯文字：每个表格行一行，同一行的单元格用制表符分隔"""
    return "\n".join("\t".join(ln.text.strip() for ln in row) for row in rows)


def table_grid(lines: Sequence[OcrLine], page_width: float) -> TableGrid | None:
    """粗略行列网格。行按 y 中心聚类，列按行的 x 起点一维聚类；对不齐时返回 None：
    少于 2 行或 2 列、某一列只在不到 TABLE_COL_FILL_MIN 的行里有内容（起点散乱，多半是居中对齐或跨列单元格）。
    同一格里有多段文字（一格被检测成几段）时按顺序拼接。"""
    rows = visual_rows(lines)
    if len(rows) < 2:
        return None
    flat = [ln for row in rows for ln in row]
    starts = [ln.bbox[0] for ln in flat]
    cols = _cluster_1d(starts, TABLE_COL_GAP * page_width)
    if len(cols) < 2:
        return None
    col_of = {}
    for c, members in enumerate(cols):
        for i in members:
            col_of[i] = c
    cells: dict[tuple[int, int], list[OcrLine]] = {}
    k = 0
    for r, row in enumerate(rows):
        for ln in row:
            cells.setdefault((r, col_of[k]), []).append(ln)
            k += 1
    for c in range(len(cols)):
        filled = sum(1 for r in range(len(rows)) if (r, c) in cells)
        if filled < TABLE_COL_FILL_MIN * len(rows):
            return None
    out = tuple(
        TableCell(r, c, join_lines([ln.text for ln in lns]), _union([ln.bbox for ln in lns]))
        for (r, c), lns in sorted(cells.items())
    )
    return TableGrid(len(rows), len(cols), out)


# ---------- 区域聚合（doc_figure、无归属的行） ----------


def group_regions(lines: Sequence[OcrLine]) -> list[list[OcrLine]]:
    """两行朝向相同、堆叠方向的间距小于粗细中位数的 GROUP_GAP_RATIO 倍、另一轴有重叠时并入同一区域（并查集）"""
    n = len(lines)
    if n == 0:
        return []
    t = median(ln.thickness for ln in lines)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    boxes = [ln.bbox for ln in lines]
    for i in range(n):
        for j in range(i + 1, n):
            if lines[i].vertical != lines[j].vertical:
                continue
            a, b = boxes[i], boxes[j]
            if lines[i].vertical:
                gap = max(a[0], b[0]) - min(a[2], b[2])
                overlap = min(a[3], b[3]) - max(a[1], b[1])
            else:
                gap = max(a[1], b[1]) - min(a[3], b[3])
                overlap = min(a[2], b[2]) - max(a[0], b[0])
            if gap < GROUP_GAP_RATIO * t and overlap > 0:
                parent[find(i)] = find(j)
    groups: dict[int, list[OcrLine]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(lines[i])
    # 区域排序：按区域顶边聚成视觉行（从上到下），行内横排区域从左到右、竖排区域从右到左
    regions = list(groups.values())
    boxes = [_union([ln.bbox for ln in g]) for g in regions]
    rows = _cluster_1d([b[1] for b in boxes], SAME_ROW_RATIO * t)

    def x_key(i: int) -> float:
        return -boxes[i][2] if is_vertical_block("", regions[i]) else boxes[i][0]

    return [regions[i] for row in rows for i in sorted(row, key=x_key)]


# ---------- 坐标还原 ----------


def to_original(p: Point, angle: int, upright_size: tuple[float, float]) -> Point:
    """转正后的图上的点 → 原图上的点。angle 是原图内容被顺时针转过的角度（方向分类的结果），
    upright_size 是转正后的图的 (宽, 高)"""
    x, y = p
    w, h = upright_size
    if angle == 90:
        return (h - y, x)
    if angle == 180:
        return (w - x, h - y)
    if angle == 270:
        return (y, w - x)
    return (x, y)


def to_upright(p: Point, angle: int, upright_size: tuple[float, float]) -> Point:
    """原图上的点 → 转正后的图上的点（to_original 的逆变换）"""
    x, y = p
    w, h = upright_size
    if angle == 90:
        return (y, h - x)
    if angle == 180:
        return (w - x, h - y)
    if angle == 270:
        return (w - y, x)
    return (x, y)


def order_quad(poly: Sequence[Point]) -> tuple[Point, ...]:
    """四边形的点重排为 左上、右上、右下、左下（旋转坐标后起点会变，裁行时按 p0→p1 定宽度方向）"""
    pts = list(poly)
    tl = min(pts, key=lambda p: p[0] + p[1])
    br = max(pts, key=lambda p: p[0] + p[1])
    tr = min(pts, key=lambda p: p[1] - p[0])
    bl = max(pts, key=lambda p: p[1] - p[0])
    return (tl, tr, br, bl)


def vertical_ratio(polys: Sequence[Sequence[Point]]) -> float:
    """竖长（高宽比 > VERTICAL_ASPECT）的检测框所占的比例"""
    if not polys:
        return 0.0
    lines = [OcrLine(tuple((float(x), float(y)) for x, y in p), "", 0.0) for p in polys]
    return sum(ln.vertical for ln in lines) / len(lines)


def original_size(angle: int, upright_size: tuple[float, float]) -> tuple[float, float]:
    w, h = upright_size
    return (h, w) if angle in (90, 270) else (w, h)


class _Mapper:
    """把转正后的图上的坐标逆旋转回原图并归一化"""

    def __init__(self, angle: int, upright_size: tuple[float, float]) -> None:
        self.angle = angle
        self.upright = upright_size
        self.ow, self.oh = original_size(angle, upright_size)

    def point(self, p: Point) -> list[float]:
        x, y = to_original(p, self.angle, self.upright)
        return [_norm(x, self.ow), _norm(y, self.oh)]

    def poly(self, poly: Sequence[Point]) -> list[list[float]]:
        return [self.point(p) for p in poly]

    def bbox(self, b: BBox) -> list[float]:
        pts = [self.point(p) for p in ((b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3]))]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return [min(xs), min(ys), max(xs), max(ys)]


def _norm(v: float, size: float) -> float:
    return round(min(1.0, max(0.0, v / size)), COORD_DIGITS) if size > 0 else 0.0


# ---------- 组装 ----------


def _line_out(ln: OcrLine, m: _Mapper) -> dict:
    return {"text": ln.text, "bbox": m.bbox(ln.bbox), "poly": m.poly(ln.poly), "conf": round(float(ln.conf), 4)}


def _orphan_slot(blocks: Sequence[_Block], new: _Block) -> int:
    """无归属的行聚成的段落应插在版面块序列的哪个位置：找中心最近的块，中心在它下方
    （同一高度时在它右边）就插在它后面。同一高度：两个中心的垂直距离小于较矮一方高度的一半。
    两者都是竖排时按列的顺序：在它左边就插在它后面。
    横排的段落整个在一串竖排列的上方（下方）时，插在这一串列的最前面（最后面），例如竖排页的标题"""
    if not blocks:
        return 0
    nc = _center(new.bbox)
    i = min(range(len(blocks)), key=lambda k: _dist(nc, _center(blocks[k].bbox)))
    blk = blocks[i]
    bc = _center(blk.bbox)
    new_vertical = is_vertical_block(new.label, new.lines)
    if new_vertical and is_vertical_block(blk.label, blk.lines):
        after = nc[0] < bc[0]
    else:
        nh, bh = new.bbox[3] - new.bbox[1], blk.bbox[3] - blk.bbox[1]
        after = nc[0] > bc[0] if abs(nc[1] - bc[1]) < min(nh, bh) / 2 else nc[1] > bc[1]
    slot = i + 1 if after else i
    if not new_vertical:
        def vertical_at(k: int) -> bool:
            return is_vertical_block(blocks[k].label, blocks[k].lines)

        if after:
            while slot < len(blocks) and vertical_at(slot) and new.bbox[1] >= blocks[slot].bbox[3]:
                slot += 1
        else:
            while slot > 0 and vertical_at(slot - 1) and new.bbox[3] <= blocks[slot - 1].bbox[1]:
                slot -= 1
    return slot


def _merge_orphans(blocks: list[_Block], groups: Sequence[Sequence[OcrLine]]) -> list[_Block]:
    """插入位置只参照版面块确定（不参照先插进去的段落，否则版面漏检的表格会按列串起来）；
    落在同一位置的段落保持 group_regions 的顺序（从上到下、行内从左到右）"""
    slots: dict[int, list[_Block]] = {}
    for g in groups:
        new = _Block(ORPHAN_LABEL, "paragraph", _union([ln.bbox for ln in g]), list(g))
        slots.setdefault(_orphan_slot(blocks, new), []).append(new)
    merged: list[_Block] = []
    for i in range(len(blocks) + 1):
        merged.extend(slots.get(i, []))
        if i < len(blocks):
            merged.append(blocks[i])
    return merged


def _block_text(blk: _Block, vertical: bool, page_width: float) -> tuple[str, list[OcrLine], TableGrid | None, BlockType]:
    """返回 (text, 排好序的行, 表格网格, 块类型)；块类型在段落被判为列表时改为 list"""
    if blk.type in ("figure", "formula"):
        return "", [], None, blk.type
    if blk.type == "table":
        rows = visual_rows(blk.lines)
        ordered = [ln for row in rows for ln in row]
        grid = table_grid(blk.lines, page_width)
        return table_rows_text(rows), ordered, grid, "table"
    ordered = order_lines(blk.lines, vertical)
    texts = [ln.text for ln in ordered]
    if blk.type == "paragraph" and not vertical and looks_like_list(texts):
        return join_list(texts), ordered, None, "list"
    if blk.type == "list":
        return join_list(texts), ordered, None, "list"
    return join_lines(texts), ordered, None, blk.type


def assemble_page(
    lines: Sequence[OcrLine],
    layout: Sequence[LayoutBox],
    *,
    angle: int,
    upright_size: tuple[float, float],
) -> dict:
    """doc_page 的 rotation / confidence / blocks（不含 mode、lang、timing_ms）。
    layout 必须已按阅读顺序排列（PP-DocLayoutV3 的输出顺序）"""
    m = _Mapper(angle, upright_size)
    page_w, page_h = upright_size
    blocks = [_Block(b.label, block_type(b.label), b.bbox) for b in layout if b.label not in IGNORED_LABELS]
    lines = [OcrLine(ln.poly, clean_text(ln.text), ln.conf) for ln in lines]
    lines = [ln for ln in lines if ln.text]
    owners = assign_lines(lines, [b.bbox for b in blocks])
    orphans: list[OcrLine] = []
    for ln, owner in zip(lines, owners):
        if owner is not None:
            blocks[owner].lines.append(ln)
        elif not is_noise_orphan(ln):
            orphans.append(ln)
    blocks = _merge_orphans(blocks, group_regions(orphans))

    out_blocks = []
    weighted, chars = 0.0, 0
    for blk in blocks:
        vertical = is_vertical_block(blk.label, blk.lines) if blk.type not in ("figure", "formula", "table") else False
        text, ordered, grid, btype = _block_text(blk, vertical, page_w)
        for ln in ordered:
            n = len(ln.text.strip())
            weighted += ln.conf * n
            chars += n
        order = len(out_blocks)
        out_blocks.append(
            {
                "id": order,
                "type": btype,
                "source_label": blk.label,
                "order": order,
                "bbox": m.bbox(blk.bbox),
                "text": text,
                "font_size": round(median(ln.thickness for ln in ordered) / page_h, COORD_DIGITS) if ordered else 0.0,
                "vertical": vertical,
                "lines": [_line_out(ln, m) for ln in ordered],
                "table": _table_out(grid, m) if grid else None,
            }
        )
    return {
        "rotation": angle,
        "confidence": round(weighted / chars, 4) if chars else 0.0,
        "blocks": out_blocks,
    }


def _table_out(grid: TableGrid, m: _Mapper) -> dict:
    return {
        "rows": grid.rows,
        "cols": grid.cols,
        "cells": [{"r": c.r, "c": c.c, "text": c.text, "bbox": m.bbox(c.bbox)} for c in grid.cells],
    }


def assemble_figure(lines: Sequence[OcrLine], size: tuple[float, float]) -> dict:
    """doc_figure 的 has_text / text_ratio / regions。图片不旋转，size 为 (宽, 高)"""
    m = _Mapper(0, size)
    kept = [OcrLine(ln.poly, clean_text(ln.text), ln.conf) for ln in lines if ln.conf >= FIGURE_CONF_MIN]
    kept = [ln for ln in kept if ln.text]
    regions = []
    for group in group_regions(kept):
        vertical = is_vertical_block("", group)
        ordered = order_lines(group, vertical)
        regions.append(
            {
                "id": len(regions),
                "bbox": m.bbox(_union([ln.bbox for ln in group])),
                "text": join_lines([ln.text for ln in ordered]),
                "vertical": vertical,
                "lines": [_line_out(ln, m) for ln in ordered],
            }
        )
    area = size[0] * size[1]
    ratio = sum(poly_area(ln.poly) for ln in kept) / area if area else 0.0
    return {"has_text": bool(regions), "text_ratio": round(min(1.0, ratio), COORD_DIGITS), "regions": regions}


def filter_detections(
    polys: Sequence[Sequence[Point]], scores: Sequence[float], *, min_side: float, min_score: float
) -> list[int]:
    """doc_figure 的检测框过滤：短边小于 min_side 像素或分数低于 min_score 的框丢弃，返回保留的下标"""
    keep = []
    for i, (poly, score) in enumerate(zip(polys, scores)):
        if score < min_score:
            continue
        if OcrLine(tuple((float(x), float(y)) for x, y in poly), "", 0.0).thickness < min_side:
            continue
        keep.append(i)
    return keep
