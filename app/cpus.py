"""进程实际能用的 CPU 核数（不依赖 torch，文档 worker 子进程里也能用）。

onnxruntime 默认按整台机器的物理核数开线程池，不看容器的限制：docker --cpuset-cpus / --cpus 限到 8 核时，
它在 32 核的宿主机上仍开 16 个线程，线程争抢让 CPU 推理慢 3 倍以上（实测文档整页 2.9s → 9.6s）。
创建会话时显式传这里的核数。
"""

import math
import os
from pathlib import Path

CGROUP_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")


def available_cpus(affinity: int | None = None, cpu_max: Path = CGROUP_CPU_MAX) -> int:
    """CPU 亲和性掩码里的核数（--cpuset-cpus）与 cgroup v2 的 CPU 配额（--cpus，向上取整）取较小值"""
    if affinity is None:
        affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    n = affinity
    try:
        quota, period = cpu_max.read_text().split()[:2]
        if quota != "max":
            n = min(n, max(1, math.ceil(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, n)
