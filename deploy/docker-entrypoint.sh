#!/bin/sh
# 容器启动：首次启动下载模型到挂载的 /models（已存在且校验通过的会跳过），再启动服务
set -e

if [ "${USE_STUB:-1}" != "1" ]; then
    echo "[lingva] 检查模型文件（首次启动需要下载约 0.9 GB，请耐心等待）..."
    python scripts/download_models.py --model-dir "${MODEL_DIR:-/models}"
fi

# 单 worker：每个 worker 都会各自加载一份模型
exec uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000 --workers 1
