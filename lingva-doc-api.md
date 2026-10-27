# Lingva 文档识别接口（插件接入说明）

> 适用于 PDF 阅读器：扫描页整页识别 `POST /v1/doc/page`、图片识别（可擦字）`POST /v1/doc/figure`。
> 服务端只做识别，翻译仍由插件调用用户选择的翻译引擎完成。本文的示例响应都是真实服务（A10 GPU）的原始返回，
> 只截短了数组。字段定义以 `app/schemas.py` 的 `DocPageResponse`、`DocFigureResponse` 为准。

## 1. 什么时候调用

| PDF 内容 | 做法 |
| --- | --- |
| 有文本层的页面（文字可选中） | **不调用服务端**，插件直接读 PDF 文本层 |
| 扫描页、图片型 PDF（没有文本层，或文本层基本为空） | 渲染成图片，调 `/v1/doc/page` |
| 页面里的图表、截图、照片 | 调 `/v1/doc/figure`；需要在原位置显示译文时加 `erase=true` |

`/v1/doc/page` 返回的 `figure` 块（图表、图片）**不含图中文字**。需要翻译图里的文字时，按块的 `bbox` 从同一张页面图上裁出来，
再调一次 `/v1/doc/figure`（官方服务会再计一页额度，没有文字时不计）。

## 2. 启用检测

文档识别默认关闭。插件启动时调 `GET /v1/health`，`doc_pipeline.enabled` 为 `true` 才显示相关入口；
旧版本服务端没有 `doc_pipeline` 字段，按未启用处理。

```json
{
  "api_version": 1,
  "stub": false,
  "device": "cuda",
  "official": false,
  "doc_pipeline": {
    "enabled": true,
    "models": {
      "layout": "PP-DocLayoutV3",
      "det": "PP-OCRv6_medium_det",
      "orientation": "PP-LCNet_x1_0_doc_ori",
      "rec": ["PP-OCRv6_medium_rec"]
    },
    "erase": true
  }
}
```

`models.rec` 是已加载的识别模型；韩文、西里尔字母、泰文模型在第一次用到时才加载，所以一开始只有默认模型。
`erase` 表示 `/v1/doc/page` 支持返回擦字图（第 4 节）；插件可以不看它，直接带 `erase=true` 请求，
响应里没有 `erased_image` 就按旧服务端处理（旧版本会忽略不认识的表单字段）。

## 3. 通用约定

- **鉴权、错误格式与 `/v1/analyze` 完全相同**：`Authorization: Bearer <token>`。官方服务用 `/v1/credential` 申请的匿名凭证，
  自部署服务用固定的 `API_TOKEN`。错误响应统一为 `{"code": ..., "error": ..., <附加字段>}`，按 `code` 分支处理（第 6 节）。
- **请求格式**：`multipart/form-data`，图片字段名 `file`，只接受 jpg / png / webp，文件不超过 15 MB（`max_mb`）。
- **图片尺寸**：长边不超过 **2500 px**（服务端 `DOC_MAX_SIDE`，超过返回 `image_too_large` 并附 `max_side`）。
  PDF 页面建议按 **200 DPI** 渲染：A4 正好约 1654×2339；再按长边 2500 等比缩小。导出 JPEG（质量 85~90）即可，一页约 150~300 KB。
- **坐标**：全部是相对**你上传的这张图**（按 EXIF 转正后）的归一化坐标，取值 `[0, 1]`，保留 4 位小数。
  - `bbox`：`[x1, y1, x2, y2]`，左上、右下。换成像素：`x * 图宽`、`y * 图高`。
  - `poly`：文字行的四个顶点，**按文字自身的方向**排列：`p0` 是文字的左上角，`p0→p1` 是书写方向，`p1→p2` 指向文字下方。
    页面倒置时 `p0` 在原图的右下角，插件可以用 `p0→p1` 的角度把译文转成同样的方向。
- **`lang_hint`**（可选）：页面语种。决定识别模型，**请尽量传明确的语种**：

  | lang_hint | 识别模型 |
  | --- | --- |
  | `ko` | 韩文模型 |
  | `ru` `uk` `be` `bg` `sr` `mk` `kk` 等西里尔字母语言 | 西里尔模型 |
  | `th` | 泰文模型 |
  | 其他（`zh` `ja` `en` `fr` `de` … 或不传 / `auto`） | PP-OCRv6（中、日、英及拉丁字母语言） |

  只看第一段：`zh-CN`、`pt_BR` 分别按 `zh`、`pt` 处理。格式不对（如含空格）返回 `invalid_request`。
  **PP-OCRv6 识别韩文、俄文时结果完全错误，置信度却很高**，服务端无法自己发现；用户切换文档语种后要把新值传上来。
  一页里混着中日韩文字时只能选一种，另一种文字的行会识别错。
- **耗时**（A10 GPU）：整页 P50 约 0.65 秒、P95 约 1.4 秒；图片 0.1~0.5 秒（擦字再加 0.2~0.3 秒）。
  纯 CPU 部署时整页 3~8 秒。客户端超时建议 **60 秒**以上（含排队）。

## 4. `POST /v1/doc/page`：扫描页整页识别

### 请求

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `file` | 文件 | 是 | 整页图片 |
| `lang_hint` | 字符串 | 否 | 见第 3 节，最长 32 字符 |
| `erase` | 布尔 | 否 | 默认 `false`；`true` 时额外返回擦掉所有文字行后的整页图 `erased_image`。不额外计额度 |

服务端会自动判断页面方向（倒置、横置的扫描件），插件不用先转正；返回的坐标已经换算回上传的原图。

### 响应字段

| 字段 | 说明 |
| --- | --- |
| `mode` | 固定 `"doc_page"` |
| `rotation` | `0 / 90 / 180 / 270`：检测到的原图内容**顺时针**转过的角度（倒置扫描件为 180）。坐标已换算回原图，插件一般只用它来决定译文的方向 |
| `lang` | 实际使用的语种；语种模型不可用、降级到默认模型时为 `auto` |
| `rec_model` | 实际使用的识别模型 |
| `confidence` | 全页按字符数加权的平均识别置信度（0~1）。明显偏低（如 < 0.8）时可以提示用户检查语种或扫描质量 |
| `blocks[]` | 按**阅读顺序**排列的块，见下 |
| `erased_image` | `erase=true` 且页面上有文字行时：擦字后的整页图，**base64 编码的 WebP**（不带 `data:` 前缀），格式同 `/v1/doc/figure` 的 `erased_image`。**尺寸和方向与上传的图完全一致**（`rotation` 不为 0 时也是），可以直接按 `lines[].bbox` 裁取。否则为 `null` |
| `timing_ms` | 各阶段耗时（毫秒），仅供调试 |

`blocks[]` 的每一项：

| 字段 | 说明 |
| --- | --- |
| `id` | 块编号（与 `order` 相同，从 0 开始） |
| `order` | 阅读顺序，从 0 开始连续。数组本身已按这个顺序排好 |
| `type` | `heading` 标题、`paragraph` 段落、`list` 列表、`table` 表格、`figure` 图表或图片、`formula` 公式、`skip` 页眉页脚页码 |
| `source_label` | 版面模型的原始类别（`doc_title`、`text`、`header`、`chart` 等），一般不用；空串表示版面模型漏检、由零散文字行聚成的段落 |
| `bbox` | 块的范围（归一化） |
| `text` | 拼好的整段文字：英文行尾连字符已合并，中日文行间不加空格。`list` 每项一行；`table` 每行一行、单元格之间用 `\t` 分隔；`figure` 和 `formula` 为空串 |
| `font_size` | 块内文字行粗细的中位数，相对转正后页面的高度。换算成像素：`font_size × 图高`（`rotation` 为 90 / 270 时乘图宽） |
| `vertical` | 是否竖排（竖排日文、竖排中文、旋转 90° 的坐标轴标签等） |
| `lines[]` | 块内的文字行，按阅读顺序：`text`、`bbox`、`poly`、`conf`（该行的识别置信度） |
| `table` | 仅 `type=table` 且能对齐成网格时有值：`rows`、`cols`、`cells[]`（`r`、`c`、`text`、`bbox`）；否则为 `null`，按 `text` 逐行处理 |

**建议的处理方式**

- 翻译 `heading`、`paragraph`、`list`、`table`（有 `table` 时按单元格翻译，否则按 `text` 的行）。
- `skip`（页眉、页脚、页码）默认不翻译，原样显示。例外见下面的「页眉的跨页判断」。
- `formula` 不翻译。
- `figure` 不含文字，需要时裁出来调 `/v1/doc/figure`。
- 译文覆盖在 `bbox` 上，字号参考 `font_size`，竖排块按竖排排版。
- 擦字图（`erase=true`）上**所有文字行都已擦掉**，包括 `skip`、`formula` 块里的行；`figure` 块（图表、图片）内部的文字
  不在 `lines` 里，不擦。只把要翻译的块的行从擦字图上裁下来贴回页面，其余区域保留原图。擦除范围是行的 `poly`
  按行粗细往外膨胀 15%，行框外的像素不变，所以贴回时按 `bbox` 往外多放一些也不会带出别的东西。
  擦字方法同 `/v1/doc/figure`：纯色背景（包括深色横幅）直接填色，纹理背景用 LaMa；A10 上每页多 0.2~0.45 秒，
  实测整页 WebP 只有 10~40 KB。版面模型丢掉的孤立标点（如单独一行的「。」）不在 `lines` 里，也不会擦。

**页眉的跨页判断**

版面模型会把页面顶部横幅里的大标题判成页眉（`type=skip`、`source_label=header`），例如每页横幅里各不相同的
「文档翻译质量评估报告」「文本解析与结构识别规范」。直接跳过，这些标题就不会翻译。服务端每次只看一页，
分不清它和普通页眉：实测普通页眉（`Document Analysis Study / 01`）的字号同样是正文的 1.5~1.8 倍，靠字号区分不了。
插件拿得到整份文档，按「是否跨页重复」判断：

- 只看 `source_label=header` 的块；`footer`、`number` 照旧跳过。
- 文字去掉数字、空白和标点后做比较（`Document Analysis Study / 01` 和 `/ 02` 视为相同）。去掉后为空的（只有页码）跳过。
- 同样的文字在已识别的其他页上也出现过：普通页眉，跳过。没有出现过：当 `heading` 翻译。
- 只识别了一页时无从比较，先按 `heading` 翻译（页眉很短，多翻一次的代价很小）；之后识别的页发现重复，再把它改回跳过（每识别一页就对已识别的全部页重新判断一次）。

```js
// headerKey 相同的页眉块在两页以上出现 → 普通页眉
const headerKey = (text) => text.toLowerCase().replace(/[\p{N}\p{P}\p{S}\s]/gu, "");

function classifyHeaders(pages) {          // pages: 已识别页的 doc_page 响应
  const seen = new Map();                  // headerKey → 出现的页数
  for (const page of pages) {
    const keys = new Set(page.blocks.filter((b) => b.source_label === "header").map((b) => headerKey(b.text)));
    for (const k of keys) seen.set(k, (seen.get(k) ?? 0) + 1);
  }
  for (const page of pages) {
    for (const b of page.blocks) {
      if (b.source_label !== "header") continue;
      const k = headerKey(b.text);
      b.translate = k !== "" && seen.get(k) === 1;   // 只在一页出现：横幅标题，要翻译
    }
  }
}
```

### 调用示例

```bash
curl -X POST https://<服务器>/v1/doc/page \
  -H "Authorization: Bearer <token>" \
  -F "file=@page-3.jpg;type=image/jpeg" \
  -F "lang_hint=ja"   -F "erase=true"
```

```js
// 扩展里：把 PDF 页面渲染到 canvas（200 DPI，长边不超过 2500）后上传
async function recognizePage(canvas, langHint, token) {
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
  const form = new FormData();
  form.append("file", blob, "page.jpg");
  if (langHint) form.append("lang_hint", langHint);
  form.append("erase", "true"); // 需要原地替换时；旧服务端会忽略
  const resp = await fetch(`${SERVER}/v1/doc/page`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: form,
    signal: AbortSignal.timeout(60_000),
  });
  const body = await resp.json();
  if (!resp.ok) throw Object.assign(new Error(body.error), { code: body.code, status: resp.status, detail: body });
  return body; // DocPageResponse
}

// 归一化坐标 → canvas 像素
const toPx = ([x1, y1, x2, y2], w, h) => [x1 * w, y1 * h, x2 * w, y2 * h];
```

### 响应示例 1：普通扫描页（节选）

请求：200 DPI 渲染的英文扫描页（1654×2339），`lang_hint=en`。原始响应 10 KB，共 11 个块，这里保留前 2 个。

```json
{
  "mode": "doc_page",
  "rotation": 0,
  "lang": "en",
  "rec_model": "PP-OCRv6_medium_rec",
  "confidence": 0.9972,
  "blocks": [
    {
      "id": 0,
      "type": "heading",
      "source_label": "paragraph_title",
      "order": 0,
      "bbox": [0.0695, 0.0624, 0.3198, 0.0851],
      "text": "ARCHIVAL TEST/ 02",
      "font_size": 0.0167,
      "vertical": false,
      "lines": [
        {
          "text": "ARCHIVAL TEST/ 02",
          "bbox": [0.0726, 0.0628, 0.315, 0.0825],
          "poly": [[0.0726, 0.0658], [0.3144, 0.0628], [0.315, 0.0795], [0.0732, 0.0825]],
          "conf": 0.9732
        }
      ],
      "table": null
    },
    {
      "id": 1,
      "type": "paragraph",
      "source_label": "text",
      "order": 1,
      "bbox": [0.0701, 0.1103, 0.604, 0.1321],
      "text": "Research records should preserve a clear link to the source.",
      "font_size": 0.0135,
      "vertical": false,
      "lines": [
        {
          "text": "Research records should preserve a clear link to the source.",
          "bbox": [0.0732, 0.1099, 0.601, 0.1295],
          "poly": [[0.0732, 0.1159], [0.601, 0.1099], [0.601, 0.1231], [0.0738, 0.1295]],
          "conf": 0.9996
        }
      ],
      "table": null
    }
  ],
  "timing_ms": {"decode": 42, "orient": 304, "layout": 362, "ocr": 583, "assemble": 1}
}
```

这一页是轻微倾斜的扫描件，所以 `poly` 不是水平的矩形。`timing_ms` 是服务启动后第一个请求的耗时（含模型预热），
预热之后的页面一般是 `orient` 3、`layout` 约 70、`ocr` 400~700 毫秒（示例 2~4 实测）。

### 响应示例 2：页眉页脚、表格（日文页，节选）

`lang_hint=ja`。12 个块的类型和顺序如下（`text` 截短）：

| order | type | source_label | text |
| --- | --- | --- | --- |
| 0 | skip | header | 都市交通に関する調査資料／01 |
| 1 | skip | header | JAPANESE / JAPANESE |
| 2 | heading | paragraph_title | 調査概要と方法 |
| 3~7 | paragraph | text | 利用者の移動目的を分類するため、… |
| 8 | table | table | 項目\t件数\t割合\n通勤\t10\t15%\n… |
| 9 | paragraph | text | 注：本資料は架空の検証用サンプルです。 |
| 10 | skip | footer | LINGVA \| SYNTHETIC TEST MATERIAL \| … |
| 11 | skip | number | 01 |

其中的表格块（`lines` 省略，共 15 行）：

```json
{
  "id": 8,
  "type": "table",
  "source_label": "table",
  "order": 8,
  "bbox": [0.0804, 0.5956, 0.7703, 0.749],
  "text": "項目\t件数\t割合\n通勤\t10\t15%\n通学\t17\t26%\n買い物\t24\t37%\n余暇\t31\t48%",
  "vertical": false,
  "table": {
    "rows": 5,
    "cols": 3,
    "cells": [
      {"r": 0, "c": 0, "text": "項目", "bbox": [0.0919, 0.6032, 0.1306, 0.6204]},
      {"r": 0, "c": 1, "text": "件数", "bbox": [0.4111, 0.6028, 0.4504, 0.6208]},
      {"r": 0, "c": 2, "text": "割合", "bbox": [0.5955, 0.6028, 0.6354, 0.6208]},
      {"r": 1, "c": 0, "text": "通勤", "bbox": [0.0925, 0.6353, 0.1276, 0.6507]}
    ]
  }
}
```

### 响应示例 3：图表页

图里的柱状图被识别为一个 `figure` 块，`text` 和 `lines` 为空；旋转 90° 的坐标轴标题「Units」不在图表块里，单独成为一个竖排段落：

```json
[
  {"order": 0, "type": "heading", "source_label": "paragraph_title", "text": "Quarterly measurements", "vertical": false},
  {"order": 1, "type": "paragraph", "source_label": "text", "text": "All chart values are fictional; labels are part of the test corpus.", "vertical": false},
  {"order": 2, "type": "paragraph", "source_label": "", "text": "Units", "vertical": true,
   "bbox": [0.0502, 0.478, 0.0707, 0.51]},
  {"order": 3, "type": "figure", "source_label": "chart", "text": "", "lines": [], "font_size": 0.0, "vertical": false,
   "bbox": [0.1125, 0.2322, 0.9033, 0.773]},
  {"order": 4, "type": "paragraph", "source_label": "figure_title", "text": "Figure 1. All values are synthetic.", "vertical": false}
]
```

（只列出主要字段。）要翻译坐标刻度、图例，就把 `order=3` 的 `bbox` 范围从页面图上裁出来调 `/v1/doc/figure`。

### 响应示例 4：倒置的扫描页

整张图上下颠倒（PDF 本身没有旋转标记）。服务端判断出 `rotation: 180`，文字照常按阅读顺序返回，坐标仍对应上传的原图：

```json
{
  "rotation": 180,
  "blocks": [
    {
      "order": 0,
      "type": "paragraph",
      "text": "An archive report may arrive upside down.",
      "bbox": [0.4985, 0.8679, 0.8808, 0.885],
      "lines": [
        {
          "text": "An archive report may arrive upside down.",
          "bbox": [0.5021, 0.8713, 0.8772, 0.8846],
          "poly": [[0.8772, 0.8846], [0.5021, 0.8846], [0.5021, 0.8713], [0.8772, 0.8713]],
          "conf": 1.0
        }
      ]
    }
  ]
}
```

注意：第一段在原图的**下方**（倒过来之后它才在最上面），`poly[0]` 在右下，`p0→p1` 指向左，说明这行字是倒着的，
译文也应该转 180° 画上去。

## 5. `POST /v1/doc/figure`：图片识别（可擦字）

不做版面分析和方向判断，适合单张图表、截图、照片。

### 请求

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `file` | 文件 | 是 | 图片 |
| `lang_hint` | 字符串 | 否 | 同上 |
| `erase` | 布尔 | 否 | 默认 `false`；`true` 时额外返回擦掉文字后的图片 `erased_image` |

### 响应字段

| 字段 | 说明 |
| --- | --- |
| `mode` | 固定 `"doc_figure"` |
| `has_text` | 是否有文字。`false` 时 `regions` 为空、`erased_image` 为 `null`，**官方服务不扣这一页的额度** |
| `text_ratio` | 文字框面积占图片面积的比例（0~1），可用于判断「这是一张文字图还是照片」 |
| `lang` / `rec_model` | 同 `/v1/doc/page` |
| `regions[]` | 文字区域（相邻的行合在一起），按从上到下排列，同一高度从左到右（竖排区域从右到左）：`id`、`bbox`、`text`、`vertical`、`lines[]`（字段同上） |
| `erased_image` | `erase=true` 且有文字时：擦字后的整张图，**base64 编码的 WebP**（不带 `data:` 前缀，尺寸与上传的图相同），格式同 `/v1/analyze` 的 `clean_image`；否则为 `null` |
| `timing_ms` | 各阶段耗时 |

擦字：纯色背景直接填色，纹理、渐变背景用 LaMa 修补。图片较大时 `erased_image` 有几百 KB（下例 1500×1050 的照片约 500 KB）。

### 调用示例

```bash
curl -X POST https://<服务器>/v1/doc/figure \
  -H "Authorization: Bearer <token>" \
  -F "file=@chart.png;type=image/png" \
  -F "lang_hint=en" \
  -F "erase=true"
```

```js
async function recognizeFigure(blob, { langHint, erase = false, token }) {
  const form = new FormData();
  form.append("file", blob, "figure.png");
  if (langHint) form.append("lang_hint", langHint);
  form.append("erase", String(erase));
  const resp = await fetch(`${SERVER}/v1/doc/figure`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: form,
    signal: AbortSignal.timeout(60_000),
  });
  const body = await resp.json();
  if (!resp.ok) throw Object.assign(new Error(body.error), { code: body.code, status: resp.status, detail: body });
  if (body.erased_image) body.erasedUrl = `data:image/webp;base64,${body.erased_image}`;
  return body; // DocFigureResponse
}

// 从页面图上裁出 doc_page 返回的 figure 块
function cropBlock(pageCanvas, bbox) {
  const [x1, y1, x2, y2] = toPx(bbox, pageCanvas.width, pageCanvas.height);
  const c = new OffscreenCanvas(Math.round(x2 - x1), Math.round(y2 - y1));
  c.getContext("2d").drawImage(pageCanvas, x1, y1, x2 - x1, y2 - y1, 0, 0, c.width, c.height);
  return c.convertToBlob({ type: "image/png" });
}
```

注意：裁图后调用 `/v1/doc/figure`，返回的坐标是相对**裁出来的这张图**，画回页面时要加上裁剪的偏移。

### 响应示例（`erase=true`，`erased_image` 截短）

请求：一张 1500×1050 的告示照片（纸面有透视变形），`lang_hint=en`。

```json
{
  "mode": "doc_figure",
  "has_text": true,
  "text_ratio": 0.046,
  "lang": "en",
  "rec_model": "PP-OCRv6_medium_rec",
  "regions": [
    {"id": 0, "bbox": [0.1947, 0.2762, 0.5407, 0.3467], "text": "STATION NOTICE", "vertical": false, "lines": ["…"]},
    {"id": 1, "bbox": [0.1993, 0.4229, 0.356, 0.4705], "text": "Platform: 4B", "vertical": false, "lines": ["…"]},
    {"id": 2, "bbox": [0.206, 0.5533, 0.4293, 0.6067], "text": "Departures: 08:35", "vertical": false, "lines": ["…"]},
    {"id": 3, "bbox": [0.2113, 0.6781, 0.5493, 0.7305], "text": "Please stand behind the line", "vertical": false, "lines": ["…"]}
  ],
  "erased_image": "UklGRiDTBQBXRUJQVlA4IBTTBQBwagqdASrcBRoE...",
  "timing_ms": {"decode": 22, "det": 140, "rec": 71, "assemble": 0, "erase": 264}
}
```

（`lines` 的结构同 `/v1/doc/page`，这里省略。）没有文字的图片（800×600 纯色图，`erase=true`）只做检测就返回，不做识别和擦字（这次是启动后第一个请求，`det` 含预热）：

```json
{"mode": "doc_figure", "has_text": false, "text_ratio": 0.0, "lang": "en", "rec_model": "PP-OCRv6_medium_rec",
 "regions": [], "erased_image": null, "timing_ms": {"decode": 11, "det": 433}}
```

## 6. 错误码与处理建议

| code | HTTP | 附加字段 | 含义 | 插件处理 |
| --- | --- | --- | --- | --- |
| `doc_pipeline_disabled` | 503 | | 服务器没有启用文档识别 | 隐藏文档识别入口，不要重试 |
| `image_too_large` | 413 | `max_mb`；长边超限时另有 `max_side` | 文件超过 `max_mb` MB，或长边超过 `max_side` 像素 | 按 `max_side` 等比缩小（或降低 JPEG 质量）后重试 |
| `invalid_image` | 400 | | 图片无法解码或格式不支持 | 检查渲染导出的格式，不要重试 |
| `invalid_request` | 400 | `fields` | 参数缺失或格式不对（如 `lang_hint` 含非法字符） | 修正参数 |
| `unauthorized` | 401 | | 凭证无效或过期 | 官方服务：重新申请凭证后重试 |
| `banned` | 403 | | 设备或网络已被停用（官方服务） | 不要重试 |
| `quota_exceeded` | 429 | `tier`、`limit`、`reset_at`、`sponsor_limit` | 今日额度已用完（官方服务，文档与漫画共用额度） | 显示额度提示，停止本次批量识别 |
| `busy` | 503 | | 排队已满，或这台设备同时进行的文档请求已有 2 个（官方服务） | 等几秒后重试；同一设备最多同时发 2 个文档请求 |
| `worker_unavailable` | 503 | | GPU 机暂不可用（官方服务） | 稍后重试 |
| `timeout` | 504 | | 单页处理超时（默认 30 秒） | 可以重试一次；仍超时就跳过这一页 |
| `inference_failed` | 500 | | 识别失败 | 可以重试一次 |

示例：

```json
{"code": "image_too_large", "error": "Image exceeds 2500 px on the long side", "max_mb": 15, "max_side": 2500}
```

```json
{"code": "doc_pipeline_disabled", "error": "Document recognition is not enabled on this server"}
```

```json
{"code": "invalid_request", "error": "Invalid request parameters: body.lang_hint", "fields": ["body.lang_hint"]}
```

`error` 是英文说明，措辞可能调整，界面文案请按 `code` 本地化。

## 7. 官方服务的额度与并发

- **额度**：`/v1/doc/page` 每次成功计 1 页，与漫画共用每日额度。`/v1/doc/figure` 有文字时计 1 页，`has_text=false` 不计。
  任何失败（图片太大、排队已满、超时、识别失败）都不计。
- **额度响应头**：每次成功的响应带 `X-Quota-Limit`、`X-Quota-Remaining`、`X-Quota-Reset`（Unix 秒），与 `/v1/analyze` 相同。
- **并发**：同一台设备同时在处理或排队的文档请求最多 2 个，再多返回 `busy`。逐页识别一本书时，建议插件自己排队，
  同时最多发 1~2 个请求；漫画请求优先处理，文档请求在负载高时会多等一会儿。
- **自部署服务**：没有额度和设备并发限制，响应里也没有 `X-Quota-*`；排队由 `MAX_CONCURRENCY` 控制。

## 8. 接入检查清单

- [ ] 启动时读 `/v1/health` 的 `doc_pipeline.enabled`，决定是否显示「识别扫描页」入口。
- [ ] 有文本层的页面不调用服务端。
- [ ] 渲染 200 DPI、长边 ≤ 2500、JPEG 质量 85~90 上传。
- [ ] 每次都传 `lang_hint`；用户切换文档语种后用新值。
- [ ] 按 `order` 顺序翻译；`skip`、`formula` 默认不翻译；`table` 有网格时按单元格翻译。
- [ ] `source_label=header` 的块按跨页重复判断：只在一页出现的当标题翻译，重复出现的跳过（第 4 节）。
- [ ] `figure` 块需要译文时裁图调 `/v1/doc/figure`，坐标加上裁剪偏移。
- [ ] 译文按 `bbox` 定位、`font_size` 定字号；`vertical`、`rotation`（或 `poly` 的方向）决定排版方向。
- [ ] 错误按第 6 节处理：`image_too_large` 缩图重试，`busy`、`worker_unavailable`、`timeout`、`inference_failed` 稍后重试，
  `doc_pipeline_disabled` 隐藏入口，`quota_exceeded` 停止批量识别。
- [ ] 客户端超时 ≥ 60 秒，同一设备并发 ≤ 2。
