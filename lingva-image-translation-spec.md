# Lingva 漫画图片翻译 — 服务端高质量方案实施说明

> 本文档供 Claude Code 按阶段实施。每个阶段结束时停下来，向我汇报结果和验收情况，再进入下一阶段。
> 遇到文档中标注「⚠ 需确认」的地方，先停下来问我，不要自行决定。

---

## 0. 背景与目标

Lingva 是一个 Chrome 扩展（React + Vite + Ant Design，MV3）。目前图片翻译在浏览器内运行，使用的模型是：

- 文字检测：`comictextdetector.pt.onnx`（CTD）
- OCR：manga-ocr 量化 ONNX（`encoder_model_quantized.onnx`、`decoder_model_quantized.onnx`、`vocab.txt`、`generation_config.json`）
- 擦字：`migan_pipeline_v2.onnx`（MI-GAN）

这些是为了在浏览器里跑得动而选的轻量/量化模型。现在要把「看图」部分（检测、OCR、擦字）迁移到云服务器，**按质量优先重新选型**；翻译和排版仍在扩展里完成。

**职责划分**

- 服务器：接收图片 → 检测文字与气泡 → OCR → 生成擦字底图 → 返回结构化结果。无状态，不涉及翻译，不保存图片。
- 扩展：独立页面上传图片 → 调用服务器 → 用现有 LLM 批量翻译链路翻译 → 在擦字底图上叠加译文显示。

**本期明确不做**：结果缓存（服务端和扩展端都不做）、content script 网页内翻译、本地后端 / 回退链、异步任务队列。这些都是后续优化，接口设计上留好扩展余地即可。

**浏览器内的旧实现暂时保留**，不要删除，后续可能作为离线后端复用。

---

## 1. 模型选型（高质量方案）

| 环节 | 模型 | 说明 |
|---|---|---|
| 文字行检测 + 文字分割 mask | CTD（comic-text-detector），使用 PyTorch 原版权重 | 保留 CTD：它的分割 mask 是擦字质量的关键 |
| 气泡检测 | `ogkalu/comic-text-and-bubble-detector` 的完整版 `detector.onnx`（非 int8） | 提供气泡边界，用于文字分组和译文排版区域 |
| OCR | manga-ocr 全精度（`kha-white/manga-ocr-base`，PyTorch / transformers） | 替代量化版，按块识别，天然支持竖排多行 |
| 擦字 | LaMa large（manga-image-translator 的 `lama_large`，漫画微调版） | 替代 MI-GAN；GPU 上使用 bf16（不支持则 fp16，CPU 用 fp32） |
| 超分（可选） | manga-image-translator 支持的超分模型之一 | 仅在图片短边低于阈值时启用，默认关闭，Phase 4 后评估 |

**实现策略**：CTD 和 LaMa large 的推理代码、权重下载，优先复用 manga-image-translator（`zyddnys/manga-image-translator`）中的对应模块，只用它的 detector / inpainter 部分，不用它的翻译和渲染。

> ⚠ 需确认：在引入 manga-image-translator 代码（无论是作为依赖、git submodule 还是拷贝部分文件）之前，先查看它的许可证并告诉我，由我决定引入方式。

> ⚠ 需确认：气泡检测模型的输入输出格式和类别 ID 以模型仓库中的 `config.json` / `preprocessor_config.json` 为准。已知的参考信息（来自第三方镜像页面，需核实）：输入 `images` float32 `[N,3,640,640]`、RGB、/255、直接拉伸缩放，另一个输入 `orig_target_sizes` int64 `[N,2]` = `[width, height]`；输出 `labels [1,300]`、`boxes [1,300,4]`（xyxy，原图像素坐标）、`scores [1,300]`；类别 0 = 空气泡，1 = 带文字的气泡，2 = 气泡外文字。核实后在代码注释中注明来源。

所有模型版本号汇总成 `model_version` 字段返回，便于以后排查和做缓存失效。

---

## 2. 仓库与目录结构

服务端是一个新的独立目录（与扩展仓库分开，或放在扩展仓库的 `server/` 下 —— ⚠ 需确认）。

```
lingva-server/
├── app/
│   ├── main.py            # FastAPI 入口、路由、鉴权、并发控制
│   ├── config.py          # 所有配置项（环境变量）
│   ├── schemas.py         # Pydantic 请求/响应模型
│   ├── pipeline.py        # 串联各阶段，统计耗时
│   ├── device.py          # 设备与精度选择（cuda/cpu，bf16/fp16/fp32）
│   └── stages/
│       ├── decode.py      # 解码、EXIF 方向、RGB
│       ├── upscale.py     # 可选超分
│       ├── detect_text.py # CTD：文字行 + 分割 mask
│       ├── detect_bubble.py
│       ├── group.py       # 文字行 → 块（按气泡分组）+ 阅读顺序
│       ├── ocr.py         # manga-ocr
│       ├── mask.py        # 擦字 mask 生成
│       ├── inpaint.py     # LaMa large，按区域裁剪修补
│       └── color.py       # 文字前景色 / 背景色估计
├── scripts/
│   ├── download_models.py # 下载全部模型并校验 sha256
│   ├── draw_debug.py      # 把结果画到图上（块、行、气泡、顺序、mask）
│   └── bench.py           # 对测试集批量运行并输出各阶段耗时
├── tests/
│   ├── samples/           # 测试图片（不提交大图，只放 2~3 张小图）
│   ├── test_api.py        # stub 模式下的接口测试
│   └── test_group.py      # 分组/排序等纯逻辑的单元测试
├── deploy/
│   ├── lingva.service     # systemd
│   └── Caddyfile
├── models/                # 模型文件，加入 .gitignore
├── requirements.txt
├── .env.example
└── README.md
```

---

## 3. 配置项（`app/config.py`，全部来自环境变量）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `API_TOKEN` | 无（必填，stub 模式下可用 `dev-token`） | Bearer 鉴权 |
| `USE_STUB` | `1` | `1` 时不加载任何模型，返回假数据 |
| `MODEL_DIR` | `./models` | |
| `DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `MAX_UPLOAD_MB` | `15` | |
| `MAX_CONCURRENCY` | `1` | 同时进行的推理数（信号量） |
| `DETECT_SIZE` | `1536` | CTD 检测输入尺寸 |
| `BUBBLE_SCORE_MIN` | `0.3` | 气泡检测置信度阈值 |
| `MASK_DILATION` | `30` | 擦字 mask 膨胀像素（按原图尺寸），经验范围 20~40 |
| `INPAINT_SIZE` | `1024` | 单个修补区域送入 LaMa 的最大边长 |
| `INPAINT_PRECISION` | `auto` | `auto` / `bf16` / `fp16` / `fp32` |
| `UPSCALE_MIN_SIDE` | `0` | 图片短边低于此值时先超分；`0` 表示关闭 |
| `WEBP_QUALITY` | `92` | 底图编码质量 |

---

## 4. HTTP 接口契约

### `GET /v1/health`（无需鉴权）

```json
{
  "api_version": 1,
  "stub": false,
  "device": "cuda",
  "model_version": {
    "text_detector": "ctd-xxx",
    "bubble_detector": "ogkalu-ctbd-xxx",
    "ocr": "manga-ocr-base",
    "inpainter": "lama_large-xxx"
  },
  "capabilities": { "inpaint": true, "upscale": false }
}
```

### `POST /v1/analyze`

- Header：`Authorization: Bearer <token>`
- Body：`multipart/form-data`
  - `file`：图片（jpg/png/webp）
  - `inpaint`：`true` / `false`，默认 `true`
  - `debug`：`true` / `false`，默认 `false`（为 `true` 时额外返回 mask 图，便于调试）

响应：

```json
{
  "width": 1600,
  "height": 2400,
  "model_version": { "...": "同 health" },
  "blocks": [
    {
      "id": 0,
      "order": 0,
      "kind": "bubble",
      "bbox": [1320, 88, 1452, 410],
      "bubble_bbox": [1280, 40, 1500, 460],
      "vertical": true,
      "text": "なんだって…！？",
      "lines": [ { "bbox": [1400, 90, 1450, 400] } ],
      "font_size": 34,
      "fg_color": "#000000",
      "bg_color": "#ffffff"
    }
  ],
  "clean_image": "<base64 webp，inpaint=false 时为 null>",
  "debug_mask": "<base64 png，仅 debug=true 时返回>",
  "timings_ms": { "decode": 12, "detect_text": 180, "detect_bubble": 60, "group": 3, "ocr": 420, "mask": 15, "inpaint": 900, "encode": 40 }
}
```

约定：

- 所有坐标是**原图像素坐标**（即使内部做了超分或缩放，返回前都要映射回原图），格式 `[x1, y1, x2, y2]`，整数。
- `kind`：`bubble`（在气泡内的文字）或 `free`（气泡外的文字，如旁白、音效）。`free` 的 `bubble_bbox` 为 `null`。
- `order` 从 0 开始，是日漫阅读顺序（见 §5.5）。
- `lines` 只有 bbox，没有逐行文本（manga-ocr 是按块识别的）。
- `font_size` 是按原文行宽估计的原图像素字号，仅供排版参考。

错误响应体为 `{"code": "...", "error": "..."}`，`code` 是稳定的机器可读错误码，`error` 是英文说明；完整列表见 README「错误」一节。

---

## 5. 流水线详细设计（`pipeline.run`）

### 5.1 解码

Pillow 打开，按 EXIF 方向转正，转 RGB，得到 `HxWx3 uint8`。记录原图尺寸。

### 5.2 可选超分

`UPSCALE_MIN_SIDE > 0` 且短边低于阈值时，超分到短边约等于阈值后再进入后续阶段；记录缩放比例，所有输出坐标最终除以该比例映射回原图。擦字底图同样缩回原图尺寸。默认关闭。

### 5.3 文字检测（CTD）

以 `DETECT_SIZE` 运行 CTD，得到：文字行多边形/框、每行方向（横/竖）、文字分割 mask（与原图同尺寸的概率图，二值化后使用）。

### 5.4 气泡检测

运行气泡检测模型，保留 `score >= BUBBLE_SCORE_MIN` 的框，区分「带文字的气泡」和其他类别；对重叠框做 NMS（如果模型输出已去重则跳过）。

### 5.5 分组与阅读顺序（`group.py`，纯逻辑，需要单元测试）

1. 对每条文字行，计算它与每个气泡框的 IoA（交集面积 / 文字行面积）。取 IoA 最大且 ≥ 0.5 的气泡作为归属。
2. 归属于同一气泡的文字行合并为一个 `kind=bubble` 的块，`bbox` 为这些行的外接框，`bubble_bbox` 为气泡框。
3. 没有归属气泡的文字行：用 CTD / manga-image-translator 已有的文本行合并逻辑合并成块（相邻、同方向、字号相近），`kind=free`。
4. 块方向：多数行的方向。
5. 阅读顺序（日漫：从右到左、从上到下）：按块中心 y 将块分成若干横带（两块的垂直重叠超过较小块高度的 50% 视为同一横带），横带从上到下排，带内按中心 x 从右到左排。实现为可替换函数，以后可以换成按分格排序。
6. 过滤：OCR 之后移除文本为空、只有标点或长度为 1 且置信度低的块。

### 5.6 OCR（manga-ocr 全精度）

- 对每个块，按 `bbox` 外扩少量边距（如 4 px）裁剪原图，送入 manga-ocr。
- 支持批量推理（同一页的所有块一次送入），GPU 上开启半精度（如果精度无明显下降）。
- 文本做基本清洗：去掉首尾空白，统一全角省略号等。

### 5.7 擦字 mask（`mask.py`）

1. CTD 分割 mask 二值化。
2. 只保留落在「文字行框外扩若干像素」范围内的部分，避免把画面中的线条当成文字擦掉。
3. 用 `MASK_DILATION` 做膨胀，覆盖描边和毛边。
4. 对 `kind=bubble` 的块，mask 限制在气泡框内部。

### 5.8 擦字（LaMa large，`inpaint.py`）

**不要把整页缩放后一次性修补**，按区域进行，质量和速度都更好：

1. 对 mask 求连通区域，合并相距较近的区域。
2. 每个区域取外接框，四周外扩上下文边距（如区域尺寸的 50%，至少 64 px），在原图上裁剪出图块和对应 mask。
3. 图块最长边超过 `INPAINT_SIZE` 时先缩小，修补后再放大回原尺寸。尺寸对齐到模型要求的倍数（通常为 8）。
4. 用 LaMa large 修补，按 `INPAINT_PRECISION` 选择精度。
5. 把结果只在 mask 区域内贴回原图（mask 边缘做 2~3 px 羽化，避免接缝）。

### 5.9 颜色估计（`color.py`）

对每个块：前景色取文字 mask 内像素的中位数颜色；背景色取块框内、文字 mask 外像素的中位数颜色。输出 hex。

### 5.10 编码

擦字底图编码为 WebP（`WEBP_QUALITY`），base64 放入 `clean_image`。

### 5.11 耗时

每个阶段用 `time.perf_counter()` 计时，写入 `timings_ms`，同时输出到日志。

---

## 6. 服务端工程要求

- FastAPI + uvicorn，单 worker（`--workers 1`，每个 worker 都会各自加载一份模型）。
- 模型在 FastAPI `lifespan` 中加载一次。
- 推理放在 `run_in_threadpool` 中执行，外层用 `asyncio.Semaphore(MAX_CONCURRENCY)` 限流。
- `device.py`：`DEVICE=auto` 时有 CUDA 用 CUDA；`INPAINT_PRECISION=auto` 时，GPU 支持 bf16 用 bf16，否则 fp16，CPU 用 fp32。启动日志打印实际使用的设备和精度。
- `USE_STUB=1` 时不导入 torch 等重依赖，返回按图片尺寸比例生成的 2 个假块（1 个竖排 bubble、1 个 free），假底图为把块区域涂白的原图。
- `scripts/download_models.py`：从 Hugging Face / GitHub Release 下载所有模型到 `MODEL_DIR`，下载后计算 sha256 并与脚本中固定的值比对（首次运行时打印 sha256，由我确认后固定到脚本里）。
- `scripts/draw_debug.py`：输入原图和响应 JSON，输出调试图：气泡框（蓝）、块框（红）、行框（绿）、阅读顺序编号、半透明叠加的 mask。
- `scripts/bench.py`：对一个目录下的全部图片调用流水线（不经过 HTTP），输出每张图和平均的各阶段耗时。
- 本地开发环境是 **Windows**：README 中给出 Windows（PowerShell）和 Linux 两套命令；PowerShell 下测试请用 `curl.exe`（`curl` 是 `Invoke-WebRequest` 的别名）。本地没有 GPU 时，应能以 CPU 模式运行（慢但可用）。
- `requirements.txt` 中 torch 不写死 CUDA 版本，README 说明按服务器的 CUDA 版本从 PyTorch 官网选择安装命令。

---

## 7. 部署

- 目标：Linux 云服务器（⚠ 需确认：发行版、是否有 NVIDIA GPU、CUDA 版本）。
- uvicorn 只监听 `127.0.0.1:8000`。
- `deploy/lingva.service`：systemd，`EnvironmentFile` 读取 `.env`，`Restart=always`。
- `deploy/Caddyfile`：域名 `ocr.mushanyu.xyz`（⚠ 需确认），自动 HTTPS，`request_body max_size 20MB`，`reverse_proxy 127.0.0.1:8000`。
- 防火墙只开放 80 / 443。
- README 写清楚完整部署步骤。

---

## 8. 扩展端改动

### 8.1 设置

在现有设置页增加「图片识别服务」分组：`baseUrl`、`token`，存入 `chrome.storage.local` 的 `imageAnalysis` 键；提供「测试连接」按钮，调用 `/v1/health` 并显示设备、模型版本、是否支持擦字。

manifest 的 `host_permissions` 增加服务器地址（如 `https://ocr.mushanyu.xyz/*`）。

### 8.2 ImageAnalyzer 抽象（`src/lib/image-analysis/`）

```ts
// types.ts
export type BBox = [number, number, number, number];

export interface TextBlock {
  id: number;
  order: number;
  kind: 'bubble' | 'free';
  bbox: BBox;
  bubbleBbox: BBox | null;
  vertical: boolean;
  text: string;
  lines: { bbox: BBox }[];
  fontSize?: number;
  fgColor?: string;
  bgColor?: string;
}

export interface AnalysisResult {
  width: number;
  height: number;
  blocks: TextBlock[];
  cleanImageDataUrl?: string;
  debugMaskDataUrl?: string;
  modelVersion: Record<string, string>;
  timingsMs: Record<string, number>;
}

export interface AnalyzerHealth {
  device: string;
  modelVersion: Record<string, string>;
  capabilities: { inpaint: boolean; upscale: boolean };
}

export interface ImageAnalyzer {
  readonly id: string;
  healthCheck(): Promise<AnalyzerHealth>;
  analyze(image: Blob, opts?: { inpaint?: boolean; debug?: boolean; signal?: AbortSignal }): Promise<AnalysisResult>;
}

export class AnalyzerError extends Error {
  constructor(
    public code: 'UNAVAILABLE' | 'UNAUTHORIZED' | 'TOO_LARGE' | 'BAD_IMAGE' | 'NOT_CONFIGURED' | 'INTERNAL',
    message: string,
  ) { super(message); }
}
```

- `http-analyzer.ts`：`HttpAnalyzer implements ImageAnalyzer`，负责 multipart 上传、超时（默认 120 s，用 `AbortSignal.timeout` 与外部 signal 合并）、HTTP 状态码到 `AnalyzerError` 的映射、snake_case → camelCase 转换、base64 → data URL。
- `index.ts`：`getAnalyzer()` 从 `chrome.storage.local` 读取配置并创建 `HttpAnalyzer`；未配置时抛出 `NOT_CONFIGURED`。本期只有这一个实现，但上层代码只依赖 `ImageAnalyzer` 接口。

### 8.3 独立页面 `image-translate.html`

- 在 Vite 多页面配置中新增入口；从 popup 通过 `chrome.tabs.create({ url: chrome.runtime.getURL('image-translate.html') })` 打开。
- 页面是扩展上下文，**直接 import** analyzer 和现有翻译函数，不经过 background 消息。
- 布局：
  - 顶部 `Upload.Dragger`（`beforeUpload` 返回 `false`，拿到 File 自行处理），另支持 Ctrl+V 粘贴图片。
  - 左侧图片区：`Segmented` 切换视图「译文 / OCR 原文 / 原图 / 调试」。「调试」视图显示气泡框、块框、顺序编号和 mask 叠加（上传时带 `debug=true`）。
  - 右侧表格：序号、类型、原文、译文，点击行时在图上高亮对应块。
  - 状态：`idle → analyzing → translating → done`，失败进入 `error` 并显示错误码和信息；分析和翻译阶段分别显示 loading 文案和服务端返回的耗时。
- 目标语言读取扩展现有的翻译设置。

### 8.4 翻译

复用扩展现有的 LLM 批量翻译链路（sentinel 分隔格式）。按 `order` 排序后把所有块的文本作为一批送出，一个块对应一段译文；提示词中说明这是日漫对话、按阅读顺序排列、同一角色的连续气泡要保持语气一致。块数和译文数不一致时报错而不是错位显示。

### 8.5 译文排版（`MangaView` 组件）

- 底图：「译文」视图使用 `cleanImageDataUrl`，其余视图用原图。
- 所有叠加元素使用**百分比坐标**（`x / width * 100%`），图片缩放时自动对齐，无需监听 resize。
- 排版区域：`kind=bubble` 时使用 `bubbleBbox` 向内收缩 12%（气泡是椭圆，四角不可用）；`kind=free` 时使用 `bbox`。
- 竖排：`vertical=true` 时默认使用 `writing-mode: vertical-rl`（中文竖排）；设置里提供「译文始终横排」开关。
- 字号自适应：在排版区域内二分查找能完整放下译文的最大字号（上限为 `fontSize` 按显示比例换算后的值，下限 10 px），在 `useLayoutEffect` 中测量；图片显示尺寸变化时（`ResizeObserver` 观察容器）重新计算。
- 文字颜色使用 `fgColor`；没有擦字底图时（`inpaint=false` 或服务器不支持），给文字区域加 `bgColor` 半透明背景。

---

## 9. 实施阶段与验收标准

每个阶段完成后停下来汇报。

**Phase 0 — 调研（不写代码）**
阅读扩展中现有的图片翻译实现（CTD / manga-ocr / MI-GAN 的浏览器端代码）、LLM 批量翻译函数、设置页结构和 Vite 多页面配置，汇报：相关文件位置、翻译函数的签名、需要改动的扩展文件列表。确认 §1、§2、§7 中所有「⚠ 需确认」项。

**Phase 1 — 服务端骨架（stub）**
完成目录结构、配置、schemas、stub 流水线、`/v1/health`、`/v1/analyze`、鉴权、限流、`tests/test_api.py`、`deploy/` 文件、README。
验收：Windows 本地 `uvicorn` 启动；`/docs` 可访问；`curl.exe` 上传图片返回符合 §4 契约的 JSON；pytest 通过。

**Phase 2 — 扩展端对接 stub**（可与 Phase 3 起并行）
完成 §8.1、§8.2、§8.3（翻译先用原文代替），`MangaView` 先实现「OCR 原文」和「调试」视图。
验收：页面上传图片后，假块出现在正确的相对位置，缩放窗口时保持对齐；错误状态（未配置、token 错误、服务器不可达）有清晰提示。

**Phase 3 — 检测与分组**
`download_models.py`、CTD、气泡检测、`group.py`（含单元测试）、`draw_debug.py`；OCR 暂时填入「块 #n」。
验收：对 5 张以上真实漫画页输出调试图，气泡归属和阅读顺序基本正确；汇报 `timings_ms`。

**Phase 4 — OCR**
接入 manga-ocr 全精度，批量推理，文本清洗与过滤。
验收：扩展页面「OCR 原文」视图显示真实识别结果；汇报对测试页的识别情况和耗时。

**Phase 5 — 擦字**
`mask.py`、按区域的 LaMa large 修补、`color.py`、`debug_mask` 输出、`bench.py`。
验收：「调试」视图能看到 mask；擦字底图在网点、线条背景上没有明显糊斑和残留描边；汇报 CPU / GPU 下的耗时。

**Phase 6 — 翻译与排版**
§8.4、§8.5。
验收：完整链路打通，上传 → 识别 → 翻译 → 译文按气泡排版显示，竖排/横排开关生效，字号自适应不溢出。

**Phase 7 — 部署**
按 §7 部署到服务器，扩展设置填写正式地址后可正常使用。
验收：公网 HTTPS `/v1/health` 可访问；扩展端完整链路在正式服务器上跑通。

---

## 10. 通用要求

- 服务端 Python 代码使用类型标注；每个 stage 是一个可单独测试的类或函数，模型加载与推理分离。
- 不提交模型文件和大体积测试图片。
- 所有阈值、尺寸、边距等魔法数字放进 `config.py` 或 stage 顶部的常量，便于调参。
- 扩展端遵循现有代码风格和目录约定（Phase 0 调研后确定）；新增的文案同时支持现有的多语言机制（如有）。
- 不引入本文档未提到的新服务或付费依赖；需要时先问我。
