"""把 /v1/analyze 的结果画到原图上：气泡框（蓝）、块框（红）、行框（绿）、阅读顺序编号、半透明 mask。

用法：
  python scripts/draw_debug.py page.png result.json [-o out.png]
  python scripts/draw_debug.py page.png --run [-o out.png]     # 不经过 HTTP，直接调用流水线（读取 .env / 环境变量）
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BLUE, RED, GREEN, MASK = (30, 90, 255), (230, 30, 30), (0, 170, 60), (255, 0, 200)


def run_pipeline(image_path: Path) -> dict:
    from dotenv import load_dotenv

    from app.config import Settings
    from app.pipeline import load_pipeline

    load_dotenv()
    settings = Settings.from_env()
    pipeline = load_pipeline(settings)
    result = pipeline.run(image_path.read_bytes(), inpaint=False, debug=True)
    return result.model_dump()


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw(image_path: Path, result: dict) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    if img.size != (result["width"], result["height"]):
        raise SystemExit(f"图片尺寸 {img.size} 与结果 {result['width']}x{result['height']} 不一致")

    if result.get("debug_mask"):
        mask = Image.open(io.BytesIO(base64.b64decode(result["debug_mask"]))).convert("L")
        overlay = Image.new("RGB", img.size, MASK)
        img = Image.composite(Image.blend(img, overlay, 0.55), img, mask)

    d = ImageDraw.Draw(img)
    width = max(2, round(min(img.size) / 400))
    font = load_font(max(14, round(min(img.size) / 30)))
    blocks = sorted(result["blocks"], key=lambda b: b["order"])
    for b in blocks:
        if b["bubble_bbox"]:
            d.rectangle(b["bubble_bbox"], outline=BLUE, width=width)
    for b in blocks:
        for line in b["lines"]:
            d.rectangle(line["bbox"], outline=GREEN, width=max(1, width - 1))
        d.rectangle(b["bbox"], outline=RED, width=width)
    for b in blocks:
        x1, y1 = b["bbox"][:2]
        label = f"{b['order']}{'' if b['kind'] == 'bubble' else 'F'}"
        tb = d.textbbox((x1, y1), label, font=font)
        d.rectangle((tb[0] - 3, tb[1] - 2, tb[2] + 3, tb[3] + 2), fill=RED)
        d.text((x1, y1), label, fill="white", font=font)
    return img


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("result", type=Path, nargs="?", help="/v1/analyze 返回的 JSON")
    parser.add_argument("--run", action="store_true", help="直接调用流水线生成结果")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    if args.run:
        result = run_pipeline(args.image)
    elif args.result:
        result = json.loads(args.result.read_text(encoding="utf-8"))
    else:
        parser.error("需要提供结果 JSON，或使用 --run")

    out = args.output or args.image.with_name(args.image.stem + ".debug.png")
    draw(args.image, result).save(out)
    print(f"{out}  blocks={len(result['blocks'])}  timings_ms={result['timings_ms']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
