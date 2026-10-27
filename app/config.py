import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEV_TOKEN = "dev-token"
DEVICES = ("auto", "cuda", "cpu")
PRECISIONS = ("auto", "bf16", "fp16", "fp32")
DOC_ENGINES = ("onnxruntime", "paddle_static")


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    return default if raw is None or raw.strip() == "" else int(raw)


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    return default if raw is None or raw.strip() == "" else float(raw)


def _choice(env: Mapping[str, str], name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = (env.get(name) or default).strip().lower()
    if value not in allowed:
        raise ValueError(f"{name}={value!r} 无效，可选值：{', '.join(allowed)}")
    return value


@dataclass(frozen=True)
class Settings:
    api_token: str
    use_stub: bool
    model_dir: Path
    device: str
    max_upload_mb: int
    max_concurrency: int
    detect_size: int
    bubble_score_min: float
    mask_dilation: int
    inpaint_size: int
    inpaint_precision: str
    upscale_min_side: int
    webp_quality: int
    # ---- 文档识别管线（/v1/doc/page、/v1/doc/figure，见 app/doc/）----
    # 官方托管服务的前端转发文档请求时，GPU 机（本服务）也要启用
    doc_pipeline_enabled: bool = False
    # 文档 worker 子进程用的 Python，默认与服务相同（requirements-doc.txt 装在同一个环境里）
    doc_python: str = sys.executable
    doc_engine: str = "onnxruntime"
    doc_device: str = "gpu:0"
    doc_max_side: int = 2500  # 输入图片长边上限，超过返回 image_too_large（附 max_side）
    doc_det_limit_side: int = 2500  # 文字检测输入的长边上限
    doc_timeout: float = 30  # 单页处理超时（秒），超时返回 timeout 并重启 worker
    doc_start_timeout: float = 600  # worker 启动（加载模型）超时（秒）

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        use_stub = _bool(env, "USE_STUB", True)
        token = (env.get("API_TOKEN") or "").strip()
        if not token:
            if not use_stub:
                raise ValueError("API_TOKEN 未设置（仅 USE_STUB=1 时允许省略）")
            token = DEV_TOKEN
        return cls(
            api_token=token,
            use_stub=use_stub,
            model_dir=Path(env.get("MODEL_DIR") or "./models"),
            device=_choice(env, "DEVICE", "auto", DEVICES),
            max_upload_mb=_int(env, "MAX_UPLOAD_MB", 15),
            max_concurrency=max(1, _int(env, "MAX_CONCURRENCY", 1)),
            detect_size=_int(env, "DETECT_SIZE", 1536),
            bubble_score_min=_float(env, "BUBBLE_SCORE_MIN", 0.3),
            mask_dilation=_int(env, "MASK_DILATION", 30),
            inpaint_size=_int(env, "INPAINT_SIZE", 1024),
            inpaint_precision=_choice(env, "INPAINT_PRECISION", "auto", PRECISIONS),
            upscale_min_side=_int(env, "UPSCALE_MIN_SIDE", 0),
            webp_quality=_int(env, "WEBP_QUALITY", 92),
            doc_pipeline_enabled=_bool(env, "DOC_PIPELINE_ENABLED", False),
            doc_python=(env.get("DOC_PYTHON") or "").strip() or sys.executable,
            doc_engine=_choice(env, "DOC_ENGINE", "onnxruntime", DOC_ENGINES),
            doc_device=(env.get("DOC_DEVICE") or "gpu:0").strip().lower(),
            doc_max_side=max(64, _int(env, "DOC_MAX_SIDE", 2500)),
            doc_det_limit_side=max(320, _int(env, "DOC_DET_LIMIT_SIDE", 2500)),
            doc_timeout=max(1.0, _float(env, "DOC_TIMEOUT", 30)),
            doc_start_timeout=max(10.0, _float(env, "DOC_START_TIMEOUT", 600)),
        )
