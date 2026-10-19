"""下载全部模型到 MODEL_DIR 并校验 sha256。

用法：python scripts/download_models.py [--only detect ocr inpaint] [--model-dir ./models]

sha256 为 None 的文件（小的 json/txt）只打印实际 hash，确认后再固定到下面的表里；已存在时直接跳过。
已存在且 hash 一致的文件不会重复下载，可以反复运行（Docker 容器每次启动都会运行一遍）。
镜像：环境变量 HF_ENDPOINT（如 https://hf-mirror.com）、GITHUB_PROXY（GitHub 下载地址前缀代理）。
大文件的 hash 来源：GitHub Release 取自 manga-image-translator 源码中的 _MODEL_MAPPING，
Hugging Face 取自 /api/models/<repo>?blobs=true 返回的 LFS sha256；HF 文件按固定 commit 下载。
"""

import argparse
import hashlib
import http.client
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

READ_TIMEOUT = 30  # 秒：建连或连续这么久收不到数据就断开重试
RETRIES = 5
RETRY_DELAY = 5
PROGRESS_INTERVAL = 5  # 秒：非终端输出时打印进度的间隔

MIT_RELEASE ="https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3"

BUBBLE_REPO = "ogkalu/comic-text-and-bubble-detector"
BUBBLE_REV = "16e8a622f91fabc6b5b65c96d32d1183f8843546"
OCR_REPO = "kha-white/manga-ocr-base"
OCR_REV = "aa6573bd10b0d446cbf622e29c3e084914df9741"
LAMA_REPO = "dreMaz/AnimeMangaInpainting"
LAMA_REV = "2953a4e935bf01ad1471f6cbfd26ab81abeeb92d"


@dataclass(frozen=True)
class ModelFile:
    path: str
    url: str
    sha256: str | None


def hf(repo: str, rev: str, name: str, local_dir: str, sha256: str | None = None) -> ModelFile:
    return ModelFile(f"{local_dir}/{name}", f"https://huggingface.co/{repo}/resolve/{rev}/{name}", sha256)


GROUPS: dict[str, list[ModelFile]] = {
    "detect": [
        ModelFile(
            "detection/comictextdetector.pt",
            f"{MIT_RELEASE}/comictextdetector.pt",
            "1f90fa60aeeb1eb82e2ac1167a66bf139a8a61b8780acd351ead55268540cccb",
        ),
        hf(BUBBLE_REPO, BUBBLE_REV, "detector.onnx", "bubble",
           "065744e91c0594ad8663aa8b870ce3fb27222942eded5a3cc388ce23421bd195"),
        hf(BUBBLE_REPO, BUBBLE_REV, "config.json", "bubble"),
        hf(BUBBLE_REPO, BUBBLE_REV, "preprocessor_config.json", "bubble"),
    ],
    "ocr": [
        hf(OCR_REPO, OCR_REV, "pytorch_model.bin", "manga-ocr-base",
           "c63e0bb5b3ff798c5991de18a8e0956c7ee6d1563aca6729029815eda6f5c2eb"),
        hf(OCR_REPO, OCR_REV, "config.json", "manga-ocr-base"),
        hf(OCR_REPO, OCR_REV, "preprocessor_config.json", "manga-ocr-base"),
        hf(OCR_REPO, OCR_REV, "special_tokens_map.json", "manga-ocr-base"),
        hf(OCR_REPO, OCR_REV, "tokenizer_config.json", "manga-ocr-base"),
        hf(OCR_REPO, OCR_REV, "vocab.txt", "manga-ocr-base"),
    ],
    "inpaint": [
        hf(LAMA_REPO, LAMA_REV, "lama_large_512px.ckpt", "inpainting",
           "11d30fbb3000fb2eceae318b75d9ced9229d99ae990a7f8b3ac35c8d31f2c935"),
    ],
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Progress:
    """终端里用 \\r 原地刷新；输出被重定向（docker logs）时每隔几秒打印一整行，否则日志里看不到进度。"""

    def __init__(self, total: int, start: int) -> None:
        self._total = total
        self._start = start
        self._t0 = time.monotonic()
        self._last = 0.0
        self._tty = sys.stdout.isatty()

    def update(self, done: int, final: bool = False) -> None:
        now = time.monotonic()
        if not final and now - self._last < (0.5 if self._tty else PROGRESS_INTERVAL):
            return
        self._last = now
        speed = (done - self._start) / max(now - self._t0, 1e-6)
        text = f"  {done / 1e6:7.1f}"
        if self._total:
            text += f" / {self._total / 1e6:.1f} MB  {done * 100 / self._total:5.1f}%"
        else:
            text += " MB"
        text += f"  {speed / 1e6:.2f} MB/s"
        if self._total and speed > 0 and not final:
            text += f"  剩余约 {_fmt_eta((self._total - done) / speed)}"
        if self._tty:
            print("\r" + text, end="\n" if final else "", flush=True)
        else:
            print(text, flush=True)


def _fmt_eta(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


def _download_once(url: str, tmp: Path) -> None:
    """下载到 tmp；tmp 已有内容时用 Range 续传（服务器不支持就从头下）。"""
    have = tmp.stat().st_size if tmp.exists() else 0
    headers = {"User-Agent": "lingva-server"}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    # timeout 同时作用于建连和每次 read：网络卡住超过这个时间就报错重试，而不是一直挂着
    with urllib.request.urlopen(req, timeout=READ_TIMEOUT) as resp:
        if have and resp.status == 206:
            mode = "ab"
            print(f"  从 {have / 1e6:.1f} MB 处续传", flush=True)
        else:
            mode, have = "wb", 0
        length = int(resp.headers.get("Content-Length") or 0)
        total = have + length if length else 0
        progress = Progress(total, have)
        done = have
        with tmp.open(mode) as out:
            for chunk in iter(lambda: resp.read(1 << 16), b""):
                out.write(chunk)
                done += len(chunk)
                progress.update(done)
        progress.update(done, final=True)
        if total and done < total:
            raise OSError(f"连接提前断开（{done}/{total} 字节）")


def download(url: str, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, RETRIES + 1):
        try:
            _download_once(url, tmp)
            break
        except (OSError, http.client.HTTPException) as e:
            if attempt == RETRIES:
                raise
            print(f"  下载中断（{e}），{RETRY_DELAY}s 后重试（{attempt}/{RETRIES - 1}）", flush=True)
            time.sleep(RETRY_DELAY)
    tmp.replace(dest)


def resolve_url(url: str) -> str:
    """按环境变量换成镜像地址：HF_ENDPOINT（如 https://hf-mirror.com）替换 Hugging Face 域名，
    GITHUB_PROXY（形如 https://example.com/ 的前缀）加在 GitHub 下载地址前。"""
    hf_endpoint = os.environ.get("HF_ENDPOINT", "").strip().rstrip("/")
    if hf_endpoint and url.startswith("https://huggingface.co/"):
        return hf_endpoint + url[len("https://huggingface.co") :]
    github_proxy = os.environ.get("GITHUB_PROXY", "").strip()
    if github_proxy and url.startswith("https://github.com/"):
        return github_proxy.rstrip("/") + "/" + url
    return url


def fetch(model_dir: Path, item: ModelFile) -> bool:
    dest = model_dir / item.path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if item.sha256 is None:
            print(f"[skip] {item.path}（已存在）")
            return True
        if sha256_of(dest) == item.sha256:
            print(f"[skip] {item.path}（已存在，hash 一致）")
            return True
    url = resolve_url(item.url)
    print(f"[get ] {item.path}  <- {url}")
    download(url, dest)
    actual = sha256_of(dest)
    if item.sha256 is None:
        print(f"  sha256（未固定，请确认）：{actual}")
        return True
    if actual != item.sha256:
        print(f"  sha256 不匹配！期望 {item.sha256}，实际 {actual}", file=sys.stderr)
        dest.unlink()
        return False
    print("  sha256 OK")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=list(GROUPS), help="只下载指定分组")
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR") or "./models")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    ok = True
    for group in args.only or list(GROUPS):
        print(f"== {group}")
        for item in GROUPS[group]:
            ok = fetch(model_dir, item) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
