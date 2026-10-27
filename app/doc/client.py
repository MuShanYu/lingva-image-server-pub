"""主进程里管理文档 worker 子进程：启动、发送请求、超时与崩溃后重启。

分进程后，文档模型加载失败、卡死或内存泄漏不会拖垮漫画服务，超时可以直接杀掉重启。worker 一次只处理一个请求（GPU 上本来就串行），
调用方持锁排队；排队优先级由接口层的闸门决定。
"""

import itertools
import logging
import os
import queue
import subprocess
import threading
from pathlib import Path

import numpy as np

from .protocol import read_msg, write_msg
from .errors import DocInferenceError, DocTimeout

logger = logging.getLogger("lingva")

_ROOT = Path(__file__).resolve().parents[2]  # 仓库根目录：worker 以 python -m app.doc.worker 启动
_EOF = object()


class DocWorkerClient:
    def __init__(
        self,
        python: str,
        *,
        timeout: float,
        start_timeout: float,
        env: dict[str, str] | None = None,
    ) -> None:
        self._cmd = [python, "-m", "app.doc.worker"]
        self._env = {**os.environ, **(env or {}), "PYTHONPATH": str(_ROOT), "PYTHONUNBUFFERED": "1"}
        self._timeout = timeout
        self._start_timeout = start_timeout
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._proc: subprocess.Popen | None = None
        self._replies: queue.Queue = queue.Queue()
        self.models: dict = {}
        with self._lock:
            self._start()

    # ---------- 进程 ----------

    def _start(self) -> None:
        logger.info("启动文档 worker：%s", " ".join(self._cmd))
        proc = subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=self._env, cwd=_ROOT)
        replies: queue.Queue = queue.Queue()
        threading.Thread(target=self._read_loop, args=(proc, replies), daemon=True, name="doc-worker-reader").start()
        self._proc, self._replies = proc, replies
        try:
            msg = self._wait(self._start_timeout)
        except DocTimeout:
            self._kill()
            raise DocInferenceError(f"文档 worker 启动超时（{self._start_timeout:.0f}s）") from None
        if msg is _EOF or not msg[0].get("ready"):
            self._kill()
            raise DocInferenceError(f"文档 worker 启动失败（退出码 {proc.poll()}）")
        self.models = msg[0].get("models", {})

    @staticmethod
    def _read_loop(proc: subprocess.Popen, replies: queue.Queue) -> None:
        assert proc.stdout is not None
        try:
            while (msg := read_msg(proc.stdout)) is not None:
                replies.put(msg)
        except Exception:  # noqa: BLE001
            logger.exception("读取文档 worker 输出失败")
        replies.put(_EOF)

    def _wait(self, timeout: float):
        try:
            return self._replies.get(timeout=timeout)
        except queue.Empty:
            raise DocTimeout from None

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.error("文档 worker（pid %d）没有退出", proc.pid)

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ---------- 请求 ----------

    def call(self, op: str, rgb: np.ndarray | None, lang_hint: str | None, timeout: float | None = None) -> dict:
        """发送一个请求并等待结果。超时：杀掉 worker（下次调用时重启），抛出 DocTimeout；
        worker 报错或进程退出：抛出 DocInferenceError"""
        with self._lock:
            if not self.alive:
                self._start()
            assert self._proc is not None and self._proc.stdin is not None
            rid = next(self._ids)
            header: dict = {"id": rid, "op": op, "lang": lang_hint}
            payload = b""
            if rgb is not None:
                rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
                header["shape"] = list(rgb.shape)
                payload = rgb.tobytes()
            try:
                write_msg(self._proc.stdin, header, payload)
            except (BrokenPipeError, OSError) as e:
                self._kill()
                raise DocInferenceError("文档 worker 已退出") from e
            try:
                msg = self._wait(self._timeout if timeout is None else timeout)
            except DocTimeout:
                logger.error("文档 worker 处理超时（%.0fs），重启", self._timeout if timeout is None else timeout)
                self._kill()
                raise
            if msg is _EOF:
                code = self._proc.poll() if self._proc else None
                self._kill()
                raise DocInferenceError(f"文档 worker 异常退出（退出码 {code}）")
            reply, _ = msg
            if reply.get("id") != rid:  # 不应发生：一次只有一个请求在途
                self._kill()
                raise DocInferenceError("文档 worker 响应错乱")
            if not reply.get("ok"):
                raise DocInferenceError(reply.get("error") or "文档识别失败")
            if reply.get("models"):
                self.models = reply["models"]
            return reply["result"]

    def close(self) -> None:
        with self._lock:
            proc = self._proc
            if proc is not None and proc.stdin is not None:
                try:
                    proc.stdin.close()  # worker 读到 EOF 后正常退出
                    proc.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            self._kill()
