#!/bin/sh
# 容器启动：首次启动下载模型到挂载的 /models（已存在且校验通过的会跳过），再启动服务
set -e

if [ "${USE_STUB:-1}" != "1" ]; then
    echo "[lingva] 检查模型文件（首次启动需要下载约 0.9 GB，请耐心等待）..."
    python scripts/download_models.py --model-dir "${MODEL_DIR:-/models}"
    # 文档识别模型（约 300 MB，存到 /models/paddlex）。失败不影响漫画服务，文档接口会返回未启用
    # 镜像没装文档管线（构建参数 DOC_PIPELINE=0）时跳过
    if [ "${DOC_PIPELINE_ENABLED:-0}" = "1" ] && "${DOC_PYTHON:-python}" -c "import paddleocr" 2>/dev/null; then
        echo "[lingva] 检查文档识别模型..."
        "${DOC_PYTHON:-python}" scripts/download_doc_models.py || echo "[lingva] 文档识别模型下载失败"
    fi
fi

# 单 worker：每个 worker 都会各自加载一份模型
exec uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000 --workers 1
