# 第三方代码

## `ctd/` — comic-text-detector 推理代码

- 来源：[zyddnys/manga-image-translator](https://github.com/zyddnys/manga-image-translator)
  `manga_translator/detection/ctd_utils/`，commit `441d07c59a735c7db3db2e7bb8b07920afd8a9cc`
  （原始出处为 dmMaze/comic-text-detector 与 ultralytics/yolov5）
- 许可证：GPL-3.0，全文见 `LICENSE-manga-image-translator`
- 拷贝的文件：`basemodel.py`、`yolov5/common.py`、`yolov5/yolo.py`、`utils/yolov5_utils.py`、
  `utils/weight_init.py`、`utils/db_utils.py`、`utils/imgproc_utils.py`
- 修改（均以 `[lingva]` 注释标出）：
  - `yolov5/common.py`：删除未使用的 import（json、platform、requests、PIL.Image、torch.cuda.amp）
  - `utils/yolov5_utils.py`：`torchvision` 改为在 `non_max_suppression` 内延迟导入（推理流程不使用 NMS）

注意：由于包含 GPL-3.0 代码，如果对外分发本服务端代码，需要以 GPL-3.0 兼容的方式发布。
