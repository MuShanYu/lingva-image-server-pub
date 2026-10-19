import torch


def flush_denormal() -> None:
    """在调用线程上开启 denormal flush。

    CTD 的分割头/DB 头在 CPU 上会产生大量 denormal 浮点数，不开启时前向慢约 60 倍（实测 50s → 0.7s）。
    该标志按线程生效，且创建 onnxruntime 会话会把它重置，所以每次推理开始时都要在当前线程重新设置。
    """
    torch.set_flush_denormal(True)


def resolve_inpaint_precision(device: str, setting: str) -> tuple[torch.dtype | None, str]:
    """LaMa 的 autocast 精度。返回 (dtype 或 None 表示 fp32, 实际精度名)。

    LaMa 只能在 bf16 / fp32 下得到正确结果（fp16 会发暗，见 manga-image-translator 的说明），
    所以请求 fp16 或 auto 时：GPU 支持 bf16 用 bf16，否则回退 fp32；CPU 一律 fp32。
    """
    if device != "cuda" or setting == "fp32":
        return None, "fp32"
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bf16"
    return None, "fp32"


def resolve_device(setting: str) -> str:
    """DEVICE=auto 时有 CUDA 用 CUDA，否则 CPU。"""
    if setting == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda，但当前环境没有可用的 CUDA")
        return "cuda"
    if setting == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
