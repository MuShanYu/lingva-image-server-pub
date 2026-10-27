"""预下载文档识别管线的模型（版面、方向分类、检测、默认识别模型和韩 / 西里尔 / 泰文识别模型），
服务启动时就不用在线下载。需要装好 requirements-doc.txt：

  python scripts/download_doc_models.py

模型存到 PADDLE_PDX_CACHE_HOME（默认 ~/.paddlex）下的 official_models/<模型名>_onnx/，Docker 部署时挂卷保存。
国内服务器访问不了 HuggingFace 时设 PADDLE_PDX_MODEL_SOURCE=BOS。只在 CPU 上加载一次，不占显存。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    from paddleocr import DocImgOrientationClassification, LayoutDetection, TextDetection, TextRecognition

    from app.doc.labels import DEFAULT_REC, DET_MODEL, LANG_REC, LAYOUT_MODEL, ORIENT_MODEL

    # 只加载、不推理：单线程即可，也免得 onnxruntime 在限核的容器里按宿主机核数开线程、绑核失败刷屏
    kw = {"engine": "onnxruntime", "device": "cpu", "engine_config": {"intra_op_num_threads": 1}}
    jobs = [
        (DocImgOrientationClassification, ORIENT_MODEL),
        (LayoutDetection, LAYOUT_MODEL),
        (TextDetection, DET_MODEL),
        *((TextRecognition, m) for m in sorted({DEFAULT_REC, *LANG_REC.values()})),
    ]
    failed = []
    for cls, name in jobs:
        t0 = time.perf_counter()
        try:
            cls(model_name=name, **kw)
            print(f"[ok] {name}（{time.perf_counter() - t0:.1f}s）")
        except Exception as e:  # noqa: BLE001
            failed.append(name)
            print(f"[失败] {name}：{type(e).__name__}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
