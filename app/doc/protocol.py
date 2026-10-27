"""主进程 ↔ 文档 worker 子进程的消息格式：8 字节头（JSON 长度、负载长度，小端 uint32）+ JSON + 负载字节。

请求：{"id", "op": "page" | "figure", "lang", "shape": [h, w, 3]}，负载为 RGB uint8 像素。
响应：{"id", "ok": true, "result": {...}} 或 {"id", "ok": false, "error": "..."}，没有负载。
子进程启动完成后先发一条 {"ready": true, "models": {...}}。
"""

import json
import struct
from typing import BinaryIO

_HEADER = struct.Struct("<II")


def write_msg(f: BinaryIO, header: dict, payload: bytes = b"") -> None:
    body = json.dumps(header, ensure_ascii=False).encode()
    f.write(_HEADER.pack(len(body), len(payload)))
    f.write(body)
    if payload:
        f.write(payload)
    f.flush()


def _read_exact(f: BinaryIO, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def read_msg(f: BinaryIO) -> tuple[dict, bytes] | None:
    """读一条消息；对端关闭时返回 None"""
    head = _read_exact(f, _HEADER.size)
    if head is None:
        return None
    n_body, n_payload = _HEADER.unpack(head)
    body = _read_exact(f, n_body)
    payload = _read_exact(f, n_payload) if n_payload else b""
    if body is None or payload is None:
        return None
    return json.loads(body), payload
