"""文档识别管线的评估：对测试集每页调用 doc_page，输出可视化图、每页文字、汇总表和 CER
（design/文档识别管线-实现说明.md 第 8.2 节）。需要装好 requirements-doc.txt。

用法：
  python scripts/doc_ocr_eval.py tests/samples/pdf --manifest tests/doc_ocr/manifest.example.csv --out eval-out
      [--page-refs tests/samples/pdf/test_manifest.csv] [--figures tests/samples/pdf/doc_figure_manifest.csv]
      [--server http://127.0.0.1:8000 --token ...] [--pages 2] [--device gpu:0] [--det-limit-side 2500]

--manifest 按文件登记：file（PDF 或图片文件名）、lang（作为 lang_hint）、category、reference，可选 pages
（页码范围，如 1-4 或 5,6；同一文件不同页语种不同时分多行登记）。
reference 为 text_layer 时用 PDF 自带的文本层作为参考文字（只适合文本层可信的 PDF）；为文件路径（相对 manifest）
时读取人工校对文字，多页 PDF 用单独一行 ---- 分隔各页；留空则不算 CER。

--page-refs 按页登记参考文字（测试集自带的 test_manifest.csv）：file、page、reference（相对该 csv 的路径）、
scorable_cer。scorable_cer=yes 的页用这里的参考文字，算「可信 CER」；其余页仍按 --manifest 的 reference
（通常是文本层）算，只作参考，汇总里分开统计。

CER 只算正文（去掉页眉、页脚、页码），在 NFKC 归一化、去掉空白后按字符计算；阅读顺序错乱也会体现为 CER 升高。
同时输出「无序 CER」：只比较字符多重集合，不看顺序。两者之差大致就是阅读顺序带来的误差；图表这类没有标准
阅读顺序的页面看无序 CER。文本层按 PDF 内容流排序，个别 PDF（如字形侧转的页）文本层本身的顺序就不对。

--figures 评估 doc_figure（列 image、reference，路径相对该 csv）：识别文字按区域顺序拼接后算 CER。
给了 --server 时改为调用服务的 /v1/doc/figure?erase=true（完整链路，含擦字），擦字结果存为 <图>.erased.webp。

--figure-blocks：doc_page 不输出 figure 块（图表、图片）里的文字，由插件裁出来再调 doc_figure；
打开后评估也这样做，图中文字按块的位置计入 CER。参考文字包含图中文字（如坐标刻度）时应打开。

输出（<out> 下，含识别文字，不要提交）：
  <页>.jpg   块框按类型着色、标出 order，行框叠在上面
  <页>.txt   按阅读顺序的块文字
  summary.csv / figures.csv / summary.json   每页耗时、块数、行数、置信度、方向、CER
"""

import argparse
import base64
import csv
import json
import logging
import statistics
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
RENDER_DPI = 200
RENDER_MAX_SIDE = 2500
# BGR
TYPE_COLORS = {
    "heading": (0, 0, 220),
    "paragraph": (200, 120, 0),
    "list": (160, 0, 160),
    "table": (0, 150, 0),
    "figure": (0, 160, 220),
    "formula": (120, 120, 0),
    "skip": (150, 150, 150),
}


def _limit(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return img if s >= 1 else cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def load_file(path: Path) -> list[tuple[str, np.ndarray, str]]:
    """返回 [(页名, RGB 图, 文本层)]；图片的文本层为空串"""
    if path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium

        out = []
        pdf = pdfium.PdfDocument(str(path))
        try:
            for i, page in enumerate(pdf):
                img = np.asarray(page.render(scale=RENDER_DPI / 72).to_pil().convert("RGB"))
                text = page.get_textpage().get_text_range()
                out.append((f"{path.stem}_p{i + 1:02d}", _limit(img, RENDER_MAX_SIDE), text))
        finally:
            pdf.close()
        return out
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return [(path.stem, _limit(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), RENDER_MAX_SIDE), "")] if img is not None else []


def cer(hyp: str, ref: str) -> float | None:
    """字符错误率：编辑距离 / 参考长度。先做 NFKC 归一化（PDF 文本层里常有 CJK 兼容汉字、全角符号，
    与识别结果的标准写法视觉上相同），再去掉所有空白"""
    h = _norm(hyp)
    r = _norm(ref)
    if not r:
        return None
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return round(prev[-1] / len(r), 4)


def cer_unordered(hyp: str, ref: str) -> float | None:
    """无序 CER：按字符多重集合比较（漏掉的、多出的取较大者，替换只算一次），是编辑距离的下界"""
    h = Counter(_norm(hyp))
    r = Counter(_norm(ref))
    total = sum(r.values())
    if not total:
        return None
    return round(max(sum((r - h).values()), sum((h - r).values())) / total, 4)


def _norm(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def body_reference(ref: str, result: dict) -> str:
    """参考文字去掉页眉、页脚、页码：文本层按 PDF 内容流排序，这些内容的位置与阅读顺序常常不同，
    留着会让 CER 虚高。按识别结果里 skip 块的文字，在参考文字里删掉第一处相同的内容（找不到就不删）"""
    r = _norm(ref)
    for b in result["blocks"]:
        if b["type"] == "skip" and b["text"]:
            r = r.replace(_norm(b["text"]), "", 1)
    return r


def fill_figure_blocks(engine, img: np.ndarray, result: dict, lang: str | None) -> None:
    """按插件的做法补上图中文字：doc_page 不输出 figure 块里的文字，插件把块裁出来再调 doc_figure。
    这里裁出每个 figure 块（按检测到的方向转正）跑 figure()，区域文字按顺序写回块的 text"""
    from app.doc.engine import _UPRIGHT

    h, w = img.shape[:2]
    for b in result["blocks"]:
        if b["type"] != "figure":
            continue
        x1, y1, x2, y2 = (round(v * s) for v, s in zip(b["bbox"], (w, h, w, h)))
        crop = np.ascontiguousarray(img[max(0, y1):y2, max(0, x1):x2])
        if crop.size == 0:
            continue
        if result["rotation"]:
            crop = np.ascontiguousarray(cv2.rotate(crop, _UPRIGHT[result["rotation"]]))
        b["text"] = "\n".join(r["text"] for r in engine.figure(crop, lang)["regions"])


def page_text(result: dict, include_skip: bool) -> str:
    return "\n\n".join(b["text"] for b in result["blocks"] if b["text"] and (include_skip or b["type"] != "skip"))


def draw(img_rgb: np.ndarray, result: dict) -> np.ndarray:
    """在原图上画块框（按类型着色、标 order）和行框；坐标是归一化到原图的"""
    vis = np.ascontiguousarray(img_rgb[:, :, ::-1]).copy()
    h, w = vis.shape[:2]
    for b in result["blocks"]:
        for ln in b["lines"]:
            pts = np.array([[x * w, y * h] for x, y in ln["poly"]], np.int32)
            cv2.polylines(vis, [pts], True, (60, 200, 60), 1)
    for b in result["blocks"]:
        color = TYPE_COLORS.get(b["type"], (0, 0, 0))
        x1, y1, x2, y2 = (round(v * s) for v, s in zip(b["bbox"], (w, h, w, h)))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
        label = f"{b['order']} {b['type']}{' V' if b['vertical'] else ''}"
        cv2.putText(vis, label, (x1 + 4, max(y1 - 6, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return vis


def _pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return round(s[min(len(s) - 1, int(round(q * (len(s) - 1))))], 1) if s else 0.0


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.DictReader(f) if (row.get("file") or row.get("image") or "").strip()
                and not (row.get("file") or row.get("image")).startswith("#")]


def parse_pages(spec: str) -> set[int] | None:
    """'1-4' / '5,6' / '' → 页码集合（从 1 开始），空表示全部"""
    if not spec.strip():
        return None
    pages: set[int] = set()
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        pages.update(range(int(a), int(b or a) + 1))
    return pages


def read_page_refs(path: Path | None) -> dict[tuple[str, int], str]:
    """--page-refs：{(文件, 页码): 参考文字}，只取 scorable_cer=yes 且有参考文件的页"""
    if path is None:
        return {}
    refs = {}
    for row in read_csv(path):
        ref = (row.get("reference") or "").strip()
        if ref and (row.get("scorable_cer") or "").strip().lower() == "yes":
            refs[(row["file"], int(row["page"]))] = (path.parent / ref).read_text(encoding="utf-8")
    return refs


def eval_pages(engine, args, page_refs: dict) -> list[dict]:
    rows = []
    cache: dict[Path, list] = {}
    for item in read_csv(args.manifest):
        path = args.input / item["file"]
        if not path.exists():
            print(f"跳过（不存在）：{path}")
            continue
        refs: list[str] = []
        ref_kind = (item.get("reference") or "").strip()
        if ref_kind and ref_kind != "text_layer":
            refs = (args.manifest.parent / ref_kind).read_text(encoding="utf-8").split("\n----\n")
        if path not in cache:
            cache[path] = load_file(path)
        wanted = parse_pages(item.get("pages") or "")
        for i, (name, img, text_layer) in enumerate(cache[path]):
            if (wanted is not None and i + 1 not in wanted) or (args.pages and i >= args.pages):
                continue
            start = time.perf_counter()
            result = engine.page(img, item.get("lang") or None)
            wall = (time.perf_counter() - start) * 1000
            if args.figure_blocks:
                fill_figure_blocks(engine, img, result, item.get("lang") or None)
            if (item["file"], i + 1) in page_refs:
                ref, source = page_refs[(item["file"], i + 1)], "reference"
            elif ref_kind == "text_layer":
                ref, source = text_layer, "text_layer"
            else:
                ref, source = (refs[i] if i < len(refs) else ""), "manual"
            source = source if _norm(ref) else ""
            (args.out / f"{name}.txt").write_text(page_text(result, include_skip=True), encoding="utf-8")
            if not args.no_vis:
                vis = draw(img, result)
                cv2.imwrite(str(args.out / f"{name}.jpg"), cv2.resize(vis, None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 80])
            types: dict[str, int] = {}
            for b in result["blocks"]:
                types[b["type"]] = types.get(b["type"], 0) + 1
            # 正文 CER：识别结果和参考文字都不含页眉、页脚、页码
            hyp = page_text(result, include_skip=False)
            body = body_reference(ref, result) if source else ""
            row = {
                "page": name,
                "category": item.get("category", ""),
                "lang": result["lang"],
                "size": f"{img.shape[1]}x{img.shape[0]}",
                "rotation": result["rotation"],
                "blocks": len(result["blocks"]),
                "lines": sum(len(b["lines"]) for b in result["blocks"]),
                "types": " ".join(f"{k}:{v}" for k, v in sorted(types.items())),
                "confidence": result["confidence"],
                "ref_source": source,
                "cer": cer(hyp, body) if source else None,
                "cer_unordered": cer_unordered(hyp, body) if source else None,
                "wall_ms": round(wall),
                **{f"{k}_ms": v for k, v in result["timing_ms"].items()},
            }
            rows.append(row)
            shown = "-" if row["cer"] is None else f"{row['cer']}/{row['cer_unordered']}"
            print(
                f"{name:44s} rot={row['rotation']:3d} blocks={row['blocks']:3d} lines={row['lines']:3d} "
                f"conf={row['confidence']:.3f} cer={shown:>13} {source:10s} {row['wall_ms']:5d}ms"
            )
    return rows


def eval_figures(engine, args) -> list[dict]:
    client = None
    if args.server:
        import httpx

        client = httpx.Client(base_url=args.server, timeout=120,
                              headers={"Authorization": f"Bearer {args.token}"} if args.token else {})
    rows = []
    for item in read_csv(args.figures):
        path = args.figures.parent / item["image"]
        ref_path = (item.get("reference") or "").strip()
        ref = (args.figures.parent / ref_path).read_text(encoding="utf-8") if ref_path else ""
        lang = (item.get("lang") or "").strip() or None
        start = time.perf_counter()
        if client is not None:
            resp = client.post("/v1/doc/figure", files={"file": (path.name, path.read_bytes())},
                               data={"erase": "true", **({"lang_hint": lang} if lang else {})})
            resp.raise_for_status()
            result = resp.json()
        else:
            img = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            result = engine.figure(img, lang)
        wall = (time.perf_counter() - start) * 1000
        hyp = "\n".join(r["text"] for r in result["regions"])
        (args.out / f"{path.stem}.txt").write_text(hyp, encoding="utf-8")
        if result.get("erased_image"):
            (args.out / f"{path.stem}.erased.webp").write_bytes(base64.b64decode(result["erased_image"]))
        row = {
            "image": path.name,
            "has_text": result["has_text"],
            "regions": len(result["regions"]),
            "text_ratio": result["text_ratio"],
            "cer": cer(hyp, ref) if ref else None,
            "cer_unordered": cer_unordered(hyp, ref) if ref else None,
            "erased": bool(result.get("erased_image")),
            "wall_ms": round(wall),
            **{f"{k}_ms": v for k, v in result["timing_ms"].items()},
        }
        rows.append(row)
        print(f"{path.name:44s} has_text={row['has_text']} regions={row['regions']:2d} "
              f"cer={row['cer']}/{row['cer_unordered']} erased={row['erased']} {row['wall_ms']:5d}ms")
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _cer_summary(rows: list[dict]) -> dict:
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    return {
        "pages": len(rows),
        "mean_cer": _mean([r["cer"] for r in rows]),
        "mean_cer_unordered": _mean([r["cer_unordered"] for r in rows]),
        "by_category": {
            k: {"pages": len(v), "cer": _mean([r["cer"] for r in v]), "cer_unordered": _mean([r["cer_unordered"] for r in v])}
            for k, v in sorted(by_cat.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="测试集目录")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--page-refs", type=Path, help="按页登记的参考文字（scorable_cer=yes 的页算可信 CER）")
    parser.add_argument("--figures", type=Path, help="doc_figure 评估清单（列 image、reference）")
    parser.add_argument("--server", help="doc_figure 改为调用这个服务（含擦字），如 http://127.0.0.1:8000")
    parser.add_argument("--token", default="", help="--server 的 Bearer token")
    parser.add_argument("--figure-blocks", action="store_true",
                        help="figure 块按插件的做法再跑一次 doc_figure，图中文字计入 CER（耗时不计入 wall_ms）")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pages", type=int, default=0, help="每个文件最多评估的页数，0 = 全部")
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--det-limit-side", type=int, default=2500)
    parser.add_argument("--no-vis", action="store_true", help="不输出可视化图")
    args = parser.parse_args()

    from app.doc.engine import DocEngine, EngineConfig

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    engine = DocEngine(EngineConfig(device=args.device, det_limit_side=args.det_limit_side))
    print(f"模型加载 {time.perf_counter() - t0:.1f}s  {engine.models}")

    rows = eval_pages(engine, args, read_page_refs(args.page_refs))
    if not rows:
        print("没有评估任何页面")
        return 1
    _write_csv(args.out / "summary.csv", rows)
    walls = [r["wall_ms"] for r in rows[1:]] or [rows[0]["wall_ms"]]  # 去掉第一页（预热）
    summary = {
        "pages": len(rows),
        "device": args.device,
        "wall_p50_ms": _pct(walls, 0.5),
        "wall_p95_ms": _pct(walls, 0.95),
        "mean_confidence": round(statistics.mean(r["confidence"] for r in rows), 4),
        "rotations": {str(a): sum(1 for r in rows if r["rotation"] == a) for a in (0, 90, 180, 270)},
        # 可信 CER：测试集按页提供、核对过的参考文字；文本层 CER 只作参考
        "cer_reference": _cer_summary([r for r in rows if r["ref_source"] in ("reference", "manual")]),
        "cer_text_layer": _cer_summary([r for r in rows if r["ref_source"] == "text_layer"]),
    }
    if args.figures:
        figures = eval_figures(engine, args)
        if figures:
            _write_csv(args.out / "figures.csv", figures)
            summary["figures"] = {
                "images": len(figures),
                "mean_cer": _mean([r["cer"] for r in figures if r["cer"] is not None]),
                "mean_cer_unordered": _mean([r["cer_unordered"] for r in figures if r["cer_unordered"] is not None]),
                "wall_ms": [r["wall_ms"] for r in figures],
            }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
