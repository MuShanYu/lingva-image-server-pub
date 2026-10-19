FROM python:3.13-slim

# torch 单独装：默认 CPU 版；GPU 部署时由 docker-compose.gpu.yml 换成 CUDA 版的索引地址
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# 也可以从平铺目录形式的镜像安装（如 https://mirrors.aliyun.com/pytorch-wheels/cu124/），
# 这时需要同时指定 TORCH_VERSION（如 2.6.0+cu124），torch 的依赖从 PIP_INDEX_URL 安装
ARG TORCH_FIND_LINKS=
ARG TORCH_VERSION=
# pip / torch 的源请用 https：wheel 没有签名，靠 TLS 保证完整性
ARG PIP_INDEX_URL=https://pypi.org/simple
# Debian 软件源镜像（可选，如 http://mirrors.aliyun.com）。用 http：slim 镜像里还没有 ca-certificates，
# apt 会校验软件源的 GPG 签名，http 不影响完整性
ARG APT_MIRROR=
# GPU 构建设为 1：把 onnxruntime 换成 onnxruntime-gpu，气泡检测也走 CUDA。
# 固定 1.22：基于 CUDA 12 + cuDNN 9，和 torch cu124 自带的 CUDA 库一致，也能跑在 CUDA 12.2 的驱动（阿里云 GRID 535）上
ARG ONNXRUNTIME_GPU=0
ARG ONNXRUNTIME_GPU_VERSION=1.22.0

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_DIR=/models

# opencv-python-headless 依赖 libglib
RUN if [ -n "${APT_MIRROR}" ]; then \
        sed -i "s|http://deb.debian.org|${APT_MIRROR%/}|g" /etc/apt/sources.list.d/debian.sources; \
    fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN if [ -n "${TORCH_FIND_LINKS}" ]; then \
        pip install --index-url "${PIP_INDEX_URL}" --find-links "${TORCH_FIND_LINKS}" "torch==${TORCH_VERSION}"; \
    else \
        pip install --index-url "${TORCH_INDEX_URL}" torch; \
    fi
COPY requirements.txt .
RUN pip install --index-url "${PIP_INDEX_URL}" -r requirements.txt \
    && if [ "${ONNXRUNTIME_GPU}" = "1" ]; then \
        pip uninstall -y onnxruntime && pip install --index-url "${PIP_INDEX_URL}" "onnxruntime-gpu==${ONNXRUNTIME_GPU_VERSION}"; \
    fi

COPY app ./app
COPY scripts ./scripts
COPY deploy/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=4)"
ENTRYPOINT ["docker-entrypoint.sh"]
