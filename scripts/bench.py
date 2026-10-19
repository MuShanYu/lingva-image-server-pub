"""对一个目录下的全部图片调用流水线（不经过 HTTP），输出每张图和平均的各阶段耗时。

用法：
  python scripts/bench.py tests/test_case [--out debug-out] [--no-inpaint]

--out 时额外保存：<name>.clean.webp（擦字底图）、<name>.debug.png（调试图）、<name>.json（响应，不含图片）。
配置读取 .env / 环境变量（需 USE_STUB=0 才会跑真实模型）。
"""

import argparse
import base64
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _natural_key(p: Path) -> tuple:
    digits = "".join(ch for ch in p.stem if ch.isdigit())
    return (p.stem.rstrip("0123456789"), int(digits) if digits else -1, p.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, help="保存擦字底图、调试图和 JSON 的目录")
    parser.add_argument("--no-inpaint", action="store_true")
    args = parser.parse_args()

    from dotenv import load_dotenv

    from app.config import Settings
    from app.pipeline import load_pipeline
    from draw_debug import draw

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="    [log] %(message)s")
    t0 = time.perf_counter()
    pipeline = load_pipeline(Settings.from_env())
    print(f"模型加载 {time.perf_counter() - t0:.1f}s  stub={pipeline.stub} device={pipeline.device}")

    images = sorted((p for p in args.directory.iterdir() if p.suffix.lower() in IMAGE_EXTS), key=_natural_key)
    if not images:
        print(f"{args.directory} 下没有图片")
        return 1
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    # 与服务端一致：推理放在工作线程里执行
    pool = ThreadPoolExecutor(max_workers=1)
    totals: dict[str, int] = {}
    walls: list[float] = []
    for path in images:
        start = time.perf_counter()
        result = pool.submit(pipeline.run, path.read_bytes(), inpaint=not args.no_inpaint, debug=True).result()
        walls.append(time.perf_counter() - start)
        r = result.model_dump()
        for k, v in r["timings_ms"].items():
            totals[k] = totals.get(k, 0) + v
        timings = " ".join(f"{k}={v}" for k, v in r["timings_ms"].items())
        print(f"{path.name:14} {r['width']}x{r['height']:<5} blocks={len(r['blocks']):2}  total={walls[-1]:.2f}s  {timings}")

        if args.out:
            if r["clean_image"]:
                (args.out / f"{path.stem}.clean.webp").write_bytes(base64.b64decode(r["clean_image"]))
            draw(path, r).save(args.out / f"{path.stem}.debug.png")
            slim = {k: v for k, v in r.items() if k not in ("clean_image", "debug_mask")}
            (args.out / f"{path.stem}.json").write_text(json.dumps(slim, ensure_ascii=False, indent=1), encoding="utf-8")

    n = len(images)
    avg = " ".join(f"{k}={round(v / n)}" for k, v in totals.items())
    print(f"平均（{n} 张）total={sum(walls) / n:.2f}s  {avg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
