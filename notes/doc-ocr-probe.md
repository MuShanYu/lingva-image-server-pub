# 文档识别管线 · 探针结论

> 对应 `design/文档识别管线-实现说明.md` 第 2 节。日期 2026-10-23。
> 脚本：`scripts/doc_ocr_probe.py`；依赖：`requirements-doc.txt`。
> 环境：上海 GPU worker（NVIDIA L20-8Q vGPU，8192 MiB，驱动 535.183），在现有 `:cuda` 镜像上起临时容器（Python 3.13.16、onnxruntime-gpu 1.22.0、torch 2.6.0+cu124），漫画服务容器同时在跑。
> 素材：`docs/` 下 8 份合成测试 PDF 共 74 页，按 200 DPI 渲染，长边不超过 2500px。

## 结论摘要

1. **onnxruntime 引擎可行，不需要 paddlepaddle。** paddleocr 3.7.0 / paddlex 3.7.2 下，`engine="onnxruntime"` 自动改用官方的 `<模型名>_onnx` 包。方案里用到的模型都有 ONNX 版本：`PP-DocLayoutV3`、`PP-OCRv6_medium_det/rec`、`PP-LCNet_x1_0_doc_ori`，以及韩、俄、泰识别模型。
2. **方案里的参数名全部被接受。** 方案没写的 `engine_config` 是传 onnxruntime 选项的唯一入口。
3. **显存是最大的风险，必须处理。** onnxruntime 每个会话各有一个显存池，默认只增不减，几个模型的峰值会叠加起来。不处理时，单独跑文档管线就能把 8G 卡占到 7.9G，漫画模型和语种模型随后都 OOM。改成「每次 Run 后收缩显存池」后，长边 2500 时峰值 2.85G、空闲 0.68G，延迟增加约 8%。
4. **依赖冲突。** paddlex 锁定 `opencv-contrib-python==4.10.0.84`（非 headless，需要 libGL），并且按发行包名做检查，headless 版不被接受。它与现有镜像的 `opencv-python-headless 5.0` 冲突，还会把 numpy 从 2.5.4 降到 2.3.5；`paddleocr[all]` 还会引入 langchain、openai 等大量用不到的包（已改为不带 `[all]`）。最初给文档 worker 建了独立的虚拟环境；**后来改为统一成一套依赖**：漫画管线也用 numpy 2.3.5 + opencv-contrib-python 4.10，结果与原来逐像素一致（第 12 节），文档 worker 仍是独立子进程，只是用同一个 Python。
5. **`lang_hint` 路由是必须的**，不是可选优化。PP-OCRv6 识别韩文和俄文时一个字符都对不上，置信度却很高（俄文页平均 0.964），只看置信度发现不了。

## 1. 版本与参数

| 项 | 结果 |
| --- | --- |
| paddleocr / paddlex | 3.7.0 / 3.7.2 |
| onnxruntime providers | TensorRT、CUDA、CPU（用 CUDA） |
| `LayoutDetection(model_name, engine)` | 接受 |
| `PaddleOCR(text_detection_model_name, text_recognition_model_name, use_doc_orientation_classify, use_doc_unwarping, use_textline_orientation, text_det_limit_type, text_det_limit_side_len, engine)` | 全部接受 |
| `predict(img, text_det_limit_type=, text_det_limit_side_len=)` | 接受，可以逐次请求指定 |
| `engine_config` | `{"provider_options": {...}}` 透传给 CUDA EP；`session_options` 只能逐项 setattr，**不支持** `add_session_config_entry` 和 RunOptions |
| 检测参数默认值 | `thresh=0.3, box_thresh=0.6, unclip_ratio=1.5, max_side_limit=4000`；`text_rec_score_thresh=0.0`（不过滤低分行） |

## 2. LayoutDetection 结果结构

```
res["boxes"]: list of {
  cls_id: int, label: str, score: float,
  coordinate: [x1, y1, x2, y2] (int, 像素),
  order: int | None,
  polygon_points: [[x, y] × 4] (float)
}
```

- **已经按阅读顺序排列**：74 页中每一页，`boxes` 列表顺序都与 `order` 字段一致（单调递增）。
- **`order` 只有文字类块才有值**：`text`、`paragraph_title`、`number`、`algorithm` 有值；`header`、`footer`、`table`、`image`、`chart`、`figure_title`、`vision_footnote` 为 None，但它们在列表里的位置是按版面插好的。所以按方案的做法，**直接用列表下标作为 `order`**，不用 `order` 字段。
- 多边形 `polygon_points` 是 4 个点；`bbox` 取 `coordinate` 即可。
- 双栏和双页跨页扫描的阅读顺序都正确，见第 8 节（测试集 v2）。

## 3. PP-DocLayoutV3 完整 label 集合（25 个，来自模型的 `inference.yml`）

`abstract, algorithm, aside_text, chart, content, display_formula, doc_title, figure_title, footer, footer_image, footnote, formula_number, header, header_image, image, inline_formula, number, paragraph_title, reference, reference_content, seal, table, text, vertical_text, vision_footnote`

修正后的 4.2 映射表：

| label | → type | 与方案的差异 |
| --- | --- | --- |
| `doc_title`、`paragraph_title` | heading | |
| `text`、`abstract`、`content`、`reference`、`reference_content`、`footnote`、`aside_text`、`figure_title`、`vision_footnote`、`algorithm` | paragraph | 新增 `reference_content`、`vision_footnote`（图表下方的注释）；**没有 `table_title`、`chart_title`**，表题和图题都归 `figure_title` |
| `vertical_text` | paragraph，且 `vertical=true` | 新增 label，直接作为竖排判定依据，不用再按行高宽比推断 |
| `table` | table | |
| `image`、`chart`、`seal`、`header_image`、`footer_image` | figure | |
| `display_formula`、`formula_number` | formula | **没有 `formula`** |
| `header`、`footer`、`number` | skip | |
| `inline_formula` | 忽略 | |
| —— | list | **没有 `list` label**。改为在组装时判断：`paragraph` 块的多数行以项目符号或编号开头，就改判为 `list` |

74 页实际出现的 label：footer 70、text 68、number 64、paragraph_title 64、header 59、table 44、figure_title 13、image 11、chart 8、vision_footnote 5、algorithm 4。

## 4. PaddleOCR 结果结构与坐标系

```
rec_texts: [str], rec_scores: [float], rec_polys: [[[x,y]×4]], rec_boxes: [[x1,y1,x2,y2]],
dt_polys, textline_orientation_angles, text_det_params, text_type, model_settings,
doc_preprocessor_res: { angle: int, output_img, rot_img, input_img, model_settings }
```

- **所有坐标都是相对于转正后的图（`doc_preprocessor_res.output_img`）**，不是原图。把第 1 页分别旋转 90°（顺时针）、180°、90°（逆时针）后识别，匹配上的 42 行与直立页坐标的中位距离只有 1.0 / 1.4 / 1.0 px，与原图坐标的距离在 600px 以上。所以第 ⑧ 步必须自己做逆旋转。
- `angle` 的含义（实测）：输入内容被**顺时针**转了多少度。`cv2.ROTATE_90_CLOCKWISE` 得到 90，`ROTATE_180` 得到 180，`ROTATE_90_COUNTERCLOCKWISE` 得到 270，三种都检测正确。响应里的 `rotation` 可以直接用这个值。
- 版面分析要在 `output_img` 上跑，和 OCR 用同一坐标系。实现时先单独调用方向分类，转正后再分别跑版面和 OCR，这时关掉 PaddleOCR 自带的方向校正，避免同一张图分类两次。

## 5. 耗时与显存

### 5.1 单页耗时（6 页，去掉首页预热；版面 + 方向 + 检测 + 识别）

| 长边 | 未收缩 P50 / P95 | 收缩显存池 P50 / P95 |
| --- | --- | --- |
| 2000 | 763 / 1028 ms | 915 / 1118 ms |
| 2500 | 956 / 1173 ms | 1088 / 1311 ms |
| 3000 | 1175 / 1398 ms | 1322 / 1568 ms |

分模块：版面约 170 ms（与输入尺寸无关），方向分类 12–22 ms，检测在长边 2000 / 2500 时约 265 / 445 ms，其余是识别（每批 6 行）。

### 5.2 显存

每个模块单独加载，测峰值减去基线（`arena_extend_strategy=kSameAsRequested`）：

| 模块 | 加载后 | 推理峰值 |
| --- | --- | --- |
| 方向分类 | 306 MB | 330 MB |
| 版面 PP-DocLayoutV3 | 450 MB | 1389 MB（固定输入尺寸） |
| 检测 PP-OCRv6_medium_det | 364 MB | 1280→1147、1600→1587、1800→2163、2000→2484、2500→3893 MB |
| 识别 PP-OCRv6_medium_rec | 378 MB | batch 6→650、16→1388、64→5523 MB |
| 韩 / 俄 / 泰识别 | 26–70 MB | —— |

整条流程（4 个模型加载在同一进程，长边 2000 / 2500）：

| 策略 | 峰值 | 推理后常驻 | 耗时 |
| --- | --- | --- | --- |
| 默认 | 5100 / 5100 MB | 5100 MB | 基准 |
| 所有会话共用一个环境显存池 | 3108 / 4517 MB | 同峰值 | 持平 |
| **每次 Run 后收缩显存池** | **2066 / 2847 MB** | **684 MB** | +8% |
| 两者都用 | 2066 / 2847 MB | 684 MB | +8% |

- 收缩用的是 RunOptions 的 `memory.enable_memory_arena_shrinkage = "gpu:0"`。paddlex 不支持传 RunOptions，探针里的做法是包装 `ONNXRuntimeRunner._create_session`（见 `enable_arena_shrinkage()`）。正式实现沿用这个包装，并在依赖版本变化时由单元测试兜底。
- 漫画容器：空闲 1.17G，处理一页后 torch 缓存常驻约 2.5G。
- **8G 预算**：漫画约 2.5G + 文档（收缩，长边 2500）峰值 2.85G + CUDA 上下文 ≈ 6G，可以同时常驻。长边 3000 时文档峰值约 3.8G，加上漫画大图的峰值会很紧，同进程重复加载漫画模型时实测在 3000 档 OOM。
- 建议初始配置：`DOC_MAX_SIDE=2500`、`DOC_DET_LIMIT_SIDE=2500`（显存紧张时降到 2000，峰值再降约 0.8G）、识别每批 6 行（默认值）、始终开启收缩。
- 语种模型每个只占 26–70 MB，**不需要用完即卸载**，懒加载后常驻即可。

## 6. 语种识别模型

官方列表里的识别模型：`PP-OCRv6_{medium,small,tiny}_rec`、`PP-OCRv5_{server,mobile}_rec`、`{korean,cyrillic,eslav,latin,en,arabic,devanagari,el,ta,te,th}_PP-OCRv5_mobile_rec`，以及若干 v3 模型。v3 模型没有 ONNX 包。

| lang_hint | 模型 | ONNX | 实测 |
| --- | --- | --- | --- |
| ko | `korean_PP-OCRv5_mobile_rec` | 有（13 MB） | 韩文页：347 个韩文字符，平均 0.982；同一页 v6 识别出 0 个韩文字符，平均 0.784 |
| ru 等西里尔语言 | `cyrillic_PP-OCRv5_mobile_rec` | 有（7.8 MB） | 俄文页：799 个西里尔字母，平均 0.993。`eslav_PP-OCRv5_mobile_rec` 为 800 个、0.983，只覆盖东斯拉夫语言；cyrillic 覆盖面更广，选它 |
| th | `th_PP-OCRv5_mobile_rec` | 有（7.6 MB） | 能加载；**测试集没有泰文页，准确率未验证** |
| zh / ja / en | `PP-OCRv6_medium_rec` | 有（74 MB） | 日文页平均 0.999（含假名 121 字）；中文页平均 0.985；英文扫描页平均 0.996；低质量复印件（中文）平均 0.767 |

v6 识别俄文页时，34 行全部输出拉丁字母，平均置信度却有 0.964。`auto` 模式无法靠置信度兜底，插件必须尽快带上明确的 `lang_hint`。之后可以考虑在服务端按字符集做一次检查：结果里几乎没有预期字符时提示插件。

## 7. 部署相关

- **模型下载**：`PADDLE_PDX_MODEL_SOURCE=BOS` 在上海机器上可用，版面模型首次下载约 100 秒。模型缓存在 `~/.paddlex/official_models/<名字>_onnx/`，共约 300 MB。部署时挂卷，并设置 `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`，否则每次启动都会检查模型源的连通性。
- **PyPI 镜像**：在这台机器上，`mirrors.aliyun.com` 对 Python 客户端（pip / urllib）每个请求都要约 24 秒，curl 只要 0.2 秒；腾讯源和清华源都正常。pip 第一次安装就是因此卡住的。`deploy/install.sh` 已改为在阿里云 ECS 上使用腾讯云 PyPI 镜像。
- **opencv**：见摘要第 4 条。如果文档 worker 和漫画共用同一环境，需要在镜像里装 `libgl1`，并用 `opencv-contrib-python==4.10.0.84` 替换 `opencv-python-headless 5.0`，同时回归测试漫画管线。推荐给文档 worker 单独建虚拟环境。

## 8. 测试集 v2（`tests/samples/pdf/`，13 个 PDF，共 94 页）补测

v2 的中、日、韩 PDF 已嵌入字体，容器内 pdfium 可以正常渲染。下面是新增素材的结果（每个文件取前 2 页）：

| 素材 | 结果 |
| --- | --- |
| 09 真双栏 | 先左栏（标题 → 6 段），后右栏（标题 → 6 段），**没有交错**；两栏底部的 "COLUMN 1/2 END" 和页脚标为 `footer` |
| 10 双页跨页扫描 | 左页（标题、正文）→ 右页（标题、正文），**没有交错** |
| 11 竖排日文 | 正文列全部标为 `vertical_text`，顺序从右到左。**方向分类会误判**：第 1 页判 0°，置信度 0.534；第 2 页判 **90°**，置信度 0.834 |
| 12 原始倒置扫描 | 判为 180°，置信度 0.925 |
| 13 泰文 | `th_PP-OCRv5_mobile_rec`：839 个泰文字符，平均 0.979；同一页 v6 为 0 个，平均 0.591 |

对实现的影响：

- **方向校正要加置信度门槛**。正常页和倒置页的置信度都在 0.92 以上，竖排页的误判在 0.84 以下。所以分类结果不为 0° 且置信度低于 `DOC_ORIENT_MIN_SCORE`（初值 0.9）时不旋转。
- 竖排页上有不属于任何版面块的行（例如 "TEST: VERTICAL TEXT"），对应 4.5 第 3 条「不能丢行」的处理。

## 10. 测试集 v2 评估（`scripts/doc_ocr_eval.py`，94 页）

环境与第 5 节相同；漫画服务容器同时常驻（2.5G）。`DOC_DET_LIMIT_SIDE=2500`，每次推理后收缩显存池。

| 指标 | 结果 |
| --- | --- |
| 单页耗时（含解码后的全部阶段） | P50 1098 ms，P95 1394 ms |
| 全卡显存峰值（含漫画容器） | 5215 / 8192 MB |
| 方向 | 倒置扫描 4 页全部判为 180°，其余 90 页为 0° |
| 平均置信度 | 0.983 |

正文 CER：按 NFKC 归一化、去掉空白，页眉、页脚、页码不计入，参考文字取 PDF 文本层。

| 类别 | CER | 说明 |
| --- | --- | --- |
| 英文论文 / 真双栏 / 竖排日文 | 0.000 | 双栏、竖排的阅读顺序全部正确 |
| 日文 | 0.001 | |
| 俄文 | 0.006 | 差异几乎全是形状相同的拉丁 A/B 与西里尔 А/В |
| 中文（正常页） | 0.009 | 生僻字误识（爰→爱、誊→誉） |
| 混排 | 0.065 | 页面上的韩文、俄文行在 `lang_hint=en` 下识别不出，属于预期 |
| 泰文 | 0.063 | 识别模型漏掉部分声调符号（่ ้） |
| 中文字形侧转页（p3/6/9） | 约 0.78 | 见下 |
| 韩文 | 0.51 | **素材本身损坏**：v2 韩文 PDF 的字形互相重叠（字体度量问题），渲染出来不可读 |

评估中发现并已修正的问题：

- **混排页被整页旋转**：中文 p3/6/9 和复印件 p3/6/9 的标题、页眉横排，正文字形侧转 90°。方向分类判为 90°，置信度 0.92，与真正横置的页没有区别。现在分类结果为 90° / 270° 时，先在原图上检测，竖长行至少占 80% 才旋转（`rotate_vertical_min`），并复用这次的检测结果，所以不增加检测次数。
- **复印件污点**：被识别成「·」「集」「命」等单字符。无归属的行只有符号，或只有一个字符且置信度 < 0.9 时丢弃，块内的行不受影响。
- **重复标点**：识别结果里的「，，」「,，」合并为一个。
- **版面漏检的内容排序**：插入位置只参照版面块，同一位置的段落按「从上到下、行内从左到右」排列，竖排列按从右到左。版面漏检的表格单元格因此按行排列，漏检的竖排列也插回了正确的列序。

已知局限：

- **字形侧转的混排页**：列顶会多出一条横向的误检行（「虚国式公要由」），其中一列的顺序也不对。这几页的 PDF 文本层本身顺序就是乱的，CER 不可信。这类页面少见，暂不处理。
- **复印件**：版面模型漏检了表格，单元格降级为普通段落（按行排列），`table` 为空。

## 11. 集成验证（第 4–6 步，上海 GPU 机）

- **独立虚拟环境**（已废弃，见第 12 节「统一依赖」）：`python -m venv --system-site-packages /opt/doc-venv`，再装 `requirements-doc.txt`（只装 `paddleocr`、`paddlex`，不带 `[all]`；虚拟环境 523 MB，带 `[all]` 时 889 MB）。
- **端到端**：服务进程（漫画模型 + 文档 worker 子进程）启动时 worker 5.2 秒就绪。
  - 整页：双栏、倒置、竖排日文、泰文均返回 200，泰文模型按需懒加载。
  - 图片：无文字照片 101 ms 返回，不扣额度。
- **并发**：只有一个服务进程时，`MAX_CONCURRENCY=2`，6 个漫画请求和 4 个文档请求同时发出，全部返回 200，漫画请求先完成。整卡显存峰值 6569 / 8192 MB。卡上额外再跑一份漫画服务时会 OOM：文档请求返回 `inference_failed`，额度退回，worker 不崩溃。
- **擦字**：复用漫画管线的 `inpaint_image` 和常驻的 LaMa，不引入 MI-GAN。
  - 表格截图：15 个区域里 12 个直接填纯色，只有表头 3 个用了 LaMa（外圈跨过表头色带边界），表头处留有很淡的方块痕迹。
  - 带纹理背景：全部用 LaMa，修补后看不出痕迹。
  - 方案里的 `DOC_ERASE_VAR_THRESHOLD` 没有单独做：纯色判定沿用漫画管线的 `FLAT_TOL` / `FLAT_RATIO_MIN`（外圈同色像素占比），作用相同。

## 9. 测试素材的问题（v1）

- 中、日、韩 PDF 都**没有嵌入字体**。没有 CJK 字体的环境下渲染出来是空白页：容器里 pdfium 全部空白，Windows 上中日正常、韩文空白，装上 `fonts-noto-cjk` 后才正常。这一轮第 2、3、5 项用的是本机 Windows 渲染的 PNG，韩文单独在装好 Noto 字体的容器里渲染。评估脚本要么固定用带 CJK 字体的环境渲染，要么直接用插件导出的页面图。
- 缺少的类别：真正的双栏页、双页跨页扫描、竖排日文 / 中文（`04` 号文件里并没有竖排页）、倒置或横置的原始扫描件、泰文、`doc_figure` 用的图表和截图。

## 12. 测试集 v5 评估与部署配置（2026-10-26，首尔 GPU 机）

环境：阿里云 ecs.gn7i-c32g1.8xlarge（NVIDIA A10 24 GB，驱动 580，Xeon Platinum 8369B 32 vCPU，184 GB 内存），
用本分支构建的 CUDA 镜像（`DOC_PIPELINE=1`）运行完整服务；压测客户端在容器内发请求，漫画素材是 15 张真实漫画页，
文档素材取自 v5 测试集的 16 个不同版式页面。显存为整卡用量（nvidia-smi），内存为容器 cgroup 用量。

### 统一依赖

漫画管线改用 paddlex 要求的 numpy 2.3.5 + opencv-contrib-python 4.10.0.84（原为 numpy 2.5.4 + opencv-python-headless 5.0），
去掉独立虚拟环境，`requirements.txt` 与 `requirements-doc.txt` 一起安装（`pip check` 无冲突，镜像 3.75 GB → 3.66 GB）。

- 4 张图（含 3 张 15、14、7 个文字块的页面）在两套依赖下的漫画结果除耗时外完全一致，`clean_image` 逐像素相同。
- mask、residue 两步在 opencv 4.10 下每张慢约 30 ms，其余无差别。
- 202 个单元测试通过。镜像里的 starlette 较新（1.10），后台 Cookie 测试的写法据此修正（与依赖统一无关）。

### 准确率（100 页 + 3 张图）

可信 CER 只算测试集按页提供参考文字的 25 页（`scorable_cer=yes`）；`--figure-blocks` 按插件的做法把 figure 块再跑一次 doc_figure。

| 类别 | 页数 | CER | 无序 CER | 说明 |
| --- | --- | --- | --- | --- |
| 扫描 / 跨页 / 横置 90°·270° / 倒置 | 8 | 0~0.001 | 0~0.001 | 方向全部判对 |
| 图表 | 2 | 0.069 | 0 | 图中文字全部正确，只是刻度数字的顺序与参考不同 |
| 竖排中文 | 2 | 0.066 | 0.066 | 顺序正确；检测框没有覆盖列尾的「。」 |
| 混排（中日韩） | 8 | 0.076 | 0.076 | `lang_hint=auto` 时韩文行识别不出（预期） |
| 复印件 | 2 | 0.037 | 0.032 | 识别模型漏掉几个「、」「。」 |
| 中文侧转混排页 | 3 | 0.088 | 0.005 | 文字基本全对；横排标题排到了最后（见下） |
| **合计** | 25 | **0.049** | **0.033** | |

文本层 CER（53 页，只作参考）0.007；韩文修复后 0.0006。doc_figure 3 张图 CER 0。单页 P50 678 ms、P95 1097 ms。
`DOC_DET_LIMIT_SIDE=2000` 时可信 CER 0.050、文本层 0.0071，只有复印件略差（0.051），速度快 6%。

这一轮修正的问题：

- **字形侧转的横排标题识别成乱码**（04 p3/6/9）：整页按 90° 转正后标题变成竖条，按竖排转 90° 裁图正好转反了。
  竖长裁图的识别置信度低于 0.6 时再转 180° 识别一次，取置信度高的（`engine._FLIP_RETRY_SCORE`；乱码 0.20，转过来 1.00）。
- **竖排页的横排标题插在两列之间**：版面漏检的横排段落整个在一串竖排列的上方（下方）时，插在这一串列的最前面（最后面）。
- **CPU 推理在限核的容器里慢 2~3 倍**：onnxruntime 按宿主机的物理核数开线程，不看 `--cpuset-cpus` / `--cpus`。
  现在按 `app/cpus.py` 算出的可用核数设置 onnxruntime 和 torch 的线程数；4 核容器里文档单页 18.4 s → 7.3 s。

已知局限：侧转混排页转正后，横排标题变成最左边的竖条，按竖排从右到左排到了最后；这类页面少见，暂不处理。

### 资源与性能

GPU 模式（A10，`MAX_CONCURRENCY=2`，漫画 + 文档）：

| 场景 | 吞吐（/分钟） | P50 | P95 | 显存峰值 | 内存峰值 |
| --- | --- | --- | --- | --- | --- |
| 空闲 | | | | 1.7 GB | 1.3 GB |
| 文档，顺序 | 79 页 | 0.60 s | 1.40 s | 3.6 GB | 2.7 GB |
| 文档，顺序，`DOC_DET_LIMIT_SIDE=2000` | 86 页 | 0.54 s | 1.32 s | 3.1 GB | 2.8 GB |
| 漫画，顺序 | 62 张 | 0.87 s | 1.92 s | 3.3 GB | 4.0 GB |
| 漫画，2 路并发 | 103 张 | 0.98 s | 2.44 s | 3.6 GB | 4.2 GB |
| 文档，2 路并发（worker 串行） | 83 页 | 1.27 s | 2.45 s | 5.1 GB | 4.2 GB |
| 混合 4 路并发 | 130 个 | 漫画 1.44 s / 文档 2.12 s | 2.4 / 3.1 s | 5.3 GB | 4.4 GB |
| 混合 8 路并发 | 123 个 | 漫画 1.48 s / 文档 5.74 s | 3.3 / 6.6 s | 5.3 GB | 4.4 GB |
| 混合 4 路，`MAX_CONCURRENCY=1` | 74 个 | 漫画 1.50 s / 文档 4.80 s | | 5.0 GB | 4.1 GB |
| 只开漫画：空闲 / 2 路并发 | 102 张 | 1.02 s | | 1.1 / 2.8 GB | 0.7 / 2.4 GB |
| 图片识别 + 擦字，顺序 | 136 张 | 0.43 s | 0.63 s | 4.1 GB | |

- **CPU 核数**（GPU 模式，`--cpuset-cpus`）：单个请求的耗时与核数无关（2 核时文档 0.67 s、漫画 0.90 s）；
  混合 4 路并发时 32 核 130、4 核 118、2 核 94 个/分钟。
- **内存**（GPU 模式，混合 4 路并发）：容器限 8 GB、6 GB 正常；限 4 GB 时服务进程被 OOM 杀掉。
- 内存峰值是容器启动以来的最大值（这台机器的内核 6.8 不支持重置 `memory.peak`），所以后面的场景包含前面场景常驻的部分。
- 首次启动下载模型约 1.2 GB；模型已在本地时 18 秒就绪（只开漫画 8 秒）。

纯 CPU 模式（同一镜像，`DEVICE=cpu`、`DOC_DEVICE=cpu`，修正线程数之后）：

| 核数 | 文档 P50 / P95 | 漫画 P50 / P95 | 内存峰值 |
| --- | --- | --- | --- |
| 16（`--cpuset-cpus`） | 2.8 / 4.5 s | 4.3 / 9.0 s | 6.8 GB |
| 8 | 4.3 / 7.1 s | 6.2 / 12.3 s | 6.7 GB |
| 4 | 7.3 / 12.8 s | 11.1 / 20.0 s | 6.8 GB |
| 4（`--cpus 4`，配额） | 4.6 / 7.6 s | 6.6 / 11.8 s | 6.7 GB |

- 8 核混合 2 路并发：漫画 P50 13 s、P95 31 s，每分钟约 9.5 个请求。容器限 8 GB 正常，限 6 GB 时 OOM。
- 纯 CPU 时推理缓冲都在内存里，内存峰值比 GPU 模式高约 2.5 GB。

### 配置建议（README「服务器配置」）

| | CPU | 内存 | 显卡 |
| --- | --- | --- | --- |
| GPU 最低 | 2 核 | 8 GB | 8 GB 显存（整卡峰值实测 5.3 GB；之前 8 GB 的 L20 vGPU 上为 6.6 GB） |
| GPU 推荐 | 4~8 核 | 16 GB | 16~24 GB 显存 |
| 纯 CPU 最低 | 8 核 | 8 GB | |
| 纯 CPU 推荐 | 16 核 | 16 GB | |

- 磁盘至少 20 GB（CUDA 镜像解压后约 11 GB，模型约 1.2 GB）。只开漫画时内存 6 GB、显存 4 GB 即可。
- 6 GB 显存的卡没有实测：按 `MAX_CONCURRENCY=1`（−0.3 GB）和 `DOC_DET_LIMIT_SIDE=2000`（−0.5 GB）估算，整卡峰值约 4.5 GB。
