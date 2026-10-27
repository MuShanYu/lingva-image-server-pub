"""文档 worker 子进程：用文档管线的独立虚拟环境启动（python -m app.doc.worker），从 stdin 读请求、
向 stdout 写结果（格式见 protocol.py），日志写 stderr。由主进程的 DocWorkerClient 管理。

PaddleOCR / paddlex 会往 stdout 打印下载、加载信息：启动时把 stdout 的文件描述符留给协议，
再把 fd 1 指向 stderr，所有库的打印都进日志，不会混进协议。

配置从环境变量读取（主进程按 Settings 传入）：DOC_DEVICE、DOC_ENGINE、DOC_DET_LIMIT_SIDE。
DOC_WORKER_FAKE=1 时不加载模型，返回假结果（测试用）；此时 op 为 sleep 时睡 lang 秒，op 为 crash 时退出。
"""

import logging
import os
import sys
import time

import numpy as np

from .protocol import read_msg, write_msg

logger = logging.getLogger("lingva")


def _engine():
    if os.environ.get("DOC_WORKER_FAKE") == "1":
        from .service import FakeEngine

        return FakeEngine()
    from .engine import DocEngine, EngineConfig

    return DocEngine(
        EngineConfig(
            device=os.environ.get("DOC_DEVICE") or "gpu:0",
            engine=os.environ.get("DOC_ENGINE") or "onnxruntime",
            det_limit_side=int(os.environ.get("DOC_DET_LIMIT_SIDE") or 2500),
        )
    )


def main() -> int:
    proto_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    proto_in = sys.stdin.buffer
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s[doc-worker]: %(message)s")
    logging.getLogger("lingva").setLevel(logging.INFO)

    t0 = time.perf_counter()
    engine = _engine()
    logger.info("文档 worker 已就绪（%.1fs）：%s", time.perf_counter() - t0, engine.models)
    write_msg(proto_out, {"ready": True, "models": engine.models})

    while (msg := read_msg(proto_in)) is not None:
        header, payload = msg
        op = header.get("op")
        reply: dict = {"id": header.get("id")}
        try:
            if op == "sleep":  # 只用于测试
                time.sleep(float(header.get("lang") or 0))
                reply.update(ok=True, result={})
            elif op == "crash":  # 只用于测试
                return 3
            else:
                rgb = np.frombuffer(payload, np.uint8).reshape(header["shape"])
                if op == "page":
                    result = engine.page(rgb, header.get("lang"))
                elif op == "figure":
                    result = engine.figure(rgb, header.get("lang"))
                else:
                    raise ValueError(f"unknown op: {op!r}")
                reply.update(ok=True, result=result, models=engine.models)
        except Exception as e:  # noqa: BLE001
            # 只记异常类型和堆栈，不记图片内容；识别文字不会出现在异常里
            logger.exception("文档识别失败（%s）", op)
            reply.update(ok=False, error=f"{type(e).__name__}: {e}")
        write_msg(proto_out, reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
