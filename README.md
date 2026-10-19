# lingva-server

Lingva 漫画图片翻译的服务端：接收图片 → 检测文字与气泡 → OCR → 生成擦字底图 → 返回结构化结果。
无状态，不做翻译，不保存图片。翻译与排版在 Lingva 扩展中完成。

> 服务端功能完整：文字检测 + 气泡检测 + 分组 / 阅读顺序 + OCR + 擦字 + 颜色估计。
> `USE_STUB=1` 时不加载任何模型，返回按图片尺寸比例生成的假数据。

## 一键部署（推荐）

适合把服务部署到自己的 Linux 服务器（Ubuntu / Debian / CentOS 等）。服务以 Docker 容器运行，
有域名时自动启用 HTTPS。

**服务器建议**：4 核以上 CPU、8 GB 内存、15 GB 磁盘（镜像约 3 GB + 模型约 0.9 GB）。
纯 CPU 下普通漫画页每张约 4~8 秒，文字很密的页面 15~20 秒；有 NVIDIA GPU 时会快很多。

1. 在服务器上拉取代码并执行安装脚本：

   ```bash
   git clone https://github.com/MuShanYu/lingva-image-server-pub.git
   cd lingva-image-server-pub
   sudo bash deploy/install.sh
   ```

   也可以不下载源码，直接一行命令安装（部署文件下载到 `/opt/lingva-server`）：

   ```bash
   curl -fsSL https://raw.githubusercontent.com/MuShanYu/lingva-image-server-pub/main/deploy/install.sh | sudo bash
   ```

   国内服务器访问 GitHub 不稳定时，可以在 `git clone` 或 `curl` 的地址前加 GitHub 代理前缀。

   脚本会依次询问：是否使用国内镜像、域名（可留空）、是否使用 GPU（检测到时才问），然后自动：
   安装 Docker（如未安装）→ 生成访问令牌和 `.env` → 拉取预构建镜像（不在本机构建）→ 启动服务 →
   下载模型 → 等待就绪。

2. 完成后屏幕上会显示 **服务地址** 和 **访问令牌**，填到 LingVa 扩展的「设置 → 图片识别服务」，
   点「测试连接」即可。

说明：

- **有域名**：先把域名解析到服务器，并在云服务器安全组放行 80 / 443 端口；脚本会启用 Caddy 自动申请证书。
- **没有域名**：服务直接通过 `http://服务器IP:8000` 访问（需放行 8000 端口），令牌以明文传输，仅建议临时使用。
- **国内服务器**：选择使用国内镜像后，镜像从阿里云 ACR 拉取（同地域的阿里云 ECS 可设 `LINGVA_ACR_VPC=yes` 走 VPC 内网）、
  启用 HTTPS 用的 Caddy 也从 ACR 拉取，全部模型从 hf-mirror.com 下载，安装过程不再访问 Docker Hub 和 GitHub
  （获取代码除外）。只有本机构建（`LINGVA_BUILD=yes`）时需要从 Docker Hub 拉取基础镜像，失败时请在
  `/etc/docker/daemon.json` 配置镜像加速器。
- **免交互**：`curl -fsSL <安装脚本地址> | sudo LINGVA_DOMAIN=ocr.example.com LINGVA_CN_MIRROR=yes bash -s -- -y`，
  所有可用变量见脚本开头的注释（如 `LINGVA_VERSION=v1.0.0` 固定版本、`LINGVA_DIR` 安装目录）。
- **升级 / 修改配置**：重新执行安装命令即可，会保留原有令牌和参数。
- **GPU**：需要先装好 NVIDIA 驱动（`nvidia-smi` 正常）；Debian / Ubuntu 上脚本会自动安装 nvidia-container-toolkit。
- **CPU 架构**：预构建镜像只有 x86_64（amd64）；ARM 服务器请用 `LINGVA_BUILD=yes` 在本机构建（仅 CPU）。
- **常用命令**（在安装目录下，即源码目录或 `/opt/lingva-server`）：`docker compose logs -f lingva`（日志）、
  `docker compose restart`、`docker compose down`。

改过源码、需要用本地代码构建镜像时，在源码目录中执行 `sudo LINGVA_BUILD=yes bash deploy/install.sh`
（首次构建约 5~15 分钟）。
Windows / macOS 可安装 Docker Desktop 后在源码目录手动执行：复制 `.env.example` 为 `.env`，
把 `USE_STUB` 改为 0、`API_TOKEN` 改为随机字符串，然后 `docker compose up -d --build`。

### 发布流程（维护者）

1. 确认 `deploy/install.sh` 开头的仓库与镜像地址（`GITHUB_REPO` 为本公开仓库，`IMAGE_CN` 为阿里云 ACR 上海实例），
   以及本 README 中的安装命令与实际仓库一致；
2. GitHub 仓库配置（可选项不配则跳过对应步骤）：
   - Secrets `ALIYUN_REGISTRY`、`ALIYUN_NAMESPACE`、`ALIYUN_USERNAME`、`ALIYUN_PASSWORD`：同时推送到阿里云镜像仓库，
     并把 `install.sh` 里的 `IMAGE_CN` 设为对应地址；
3. 打标签发布：`git tag v1.0.0 && git push origin v1.0.0`，GitHub Actions 构建 CPU（`latest`、`v1.0.0`）和
   CUDA（`cuda`、`v1.0.0-cuda`）两个镜像；
4. 首次发布后，在 GitHub 的 Packages 设置里把 `lingva-server-pub` 包改为 Public，用户才能匿名拉取；
   阿里云 ACR 上的 `lingva-server-pub` 仓库同样要设为公开。

本项目包含 GPL-3.0 许可的第三方代码（见 `app/vendor/README.md`），整体以 GPL-3.0 发布，见 `LICENSE`。

## 目录

```
app/
  main.py          FastAPI 入口、路由、鉴权、并发控制
  config.py        配置（全部来自环境变量）
  schemas.py       请求/响应模型
  pipeline.py      流水线接口与 stub 实现
  real_pipeline.py 真实流水线（串联各阶段）
  stages/          各阶段实现（检测、分组、OCR、mask、擦字、颜色 …）
  vendor/          第三方推理代码（comic-text-detector、LaMa，见 vendor/README.md）
scripts/           模型下载、调试图、批量测速
tests/             pytest
deploy/            一键部署脚本、Docker 入口、systemd 与 Caddy 配置
Dockerfile, docker-compose*.yml
```

## 本地开发

### Windows（PowerShell）

```powershell
py -3.13 -m venv .venv
# torch 先按机器情况安装（无 GPU 用 CPU 版），再安装其余依赖
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
# 下载模型（可用 --only detect 只下载当前阶段需要的）
.\.venv\Scripts\python.exe scripts\download_models.py
Copy-Item .env.example .env   # 使用真实模型时把 USE_STUB 改为 0

# 启动（--env-file 读取 .env）
.\.venv\Scripts\uvicorn.exe app.main:create_app --factory --env-file .env --host 127.0.0.1 --port 8000 --reload
```

浏览器打开 http://127.0.0.1:8000/docs 查看接口文档。

PowerShell 中的 `curl` 是 `Invoke-WebRequest` 的别名，测试请用 `curl.exe`：

```powershell
curl.exe http://127.0.0.1:8000/v1/health
curl.exe -H "Authorization: Bearer dev-token" -F "file=@tests/samples/page.jpg" -F "debug=true" http://127.0.0.1:8000/v1/analyze
```

### Linux

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # 有 GPU 时按官网选择 CUDA 版本
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python scripts/download_models.py
cp .env.example .env
.venv/bin/uvicorn app.main:create_app --factory --env-file .env --host 127.0.0.1 --port 8000 --reload

curl http://127.0.0.1:8000/v1/health
curl -H "Authorization: Bearer dev-token" -F "file=@tests/samples/page.jpg" -F "debug=true" http://127.0.0.1:8000/v1/analyze
```

### 测试与调试

```bash
python -m pytest -q
# 直接调用流水线并输出调试图：气泡框（蓝）、块框（红，编号为阅读顺序，F = free）、行框（绿）、mask（品红）
python scripts/draw_debug.py page.png --run -o page.debug.png
# 或对已有的 /v1/analyze 响应 JSON 作图
python scripts/draw_debug.py page.png result.json
# 批量跑一个目录，输出各阶段耗时；--out 同时保存擦字底图、调试图和 JSON
python scripts/bench.py tests/test_case --out debug-out
```

## 配置

全部通过环境变量设置，说明见 `.env.example`。`USE_STUB=0` 时必须设置 `API_TOKEN`。

## 接口

- `GET /v1/health`：无需鉴权，返回 `api_version`、`stub`、`device`、`model_version`、`capabilities`。
- `POST /v1/analyze`：`Authorization: Bearer <token>`，`multipart/form-data`：
  - `file`：jpg / png / webp
  - `inpaint`：默认 `true`；`false` 时 `clean_image` 为 `null`
  - `debug`：默认 `false`；`true` 时返回 `debug_mask`（base64 PNG）

所有坐标均为 EXIF 转正后的原图像素坐标 `[x1, y1, x2, y2]`（x2、y2 不含）。

### 错误

错误统一返回 `{"code": "...", "error": "...", ...}`：

- `code`：稳定的机器可读错误码，客户端按它分支处理、显示本地化文案。已发布的 code 含义不变，只会新增。
- `error`：英文说明，仅供调试，措辞可能调整，客户端不要依赖。
- 其余字段是该错误附带的参数，客户端组装文案时代入。

| code | HTTP | 附加字段 | 含义 |
|---|---|---|---|
| `invalid_request` | 400 | `fields`（参数校验失败时） | 参数缺失或格式不对 |
| `not_found` | 404 | | 路径不存在 |
| `method_not_allowed` | 405 | | 请求方法不对 |
| `internal_error` | 500 | | 未预期的服务端异常 |
| `unauthorized` | 401 | | token 无效或缺失 |
| `invalid_image` | 400 | | 图片无法解码或格式不支持（仅 jpg / png / webp） |
| `image_too_large` | 413 | `max_mb` | 图片超过大小上限 |
| `inference_failed` | 500 | | 识别失败（服务端日志记录完整堆栈） |

完整定义见 `app/errors.py`。

## 手动部署（不用 Docker，systemd）

一般用上面的一键部署即可；需要直接部署在宿主机上时按以下步骤。以 Debian/Ubuntu 为例，服务部署在
`/opt/lingva-server`，以独立用户 `lingva` 运行。

1. 准备代码与环境

   ```bash
   sudo useradd --system --create-home --home-dir /opt/lingva-server lingva
   sudo -u lingva git clone https://github.com/MuShanYu/lingva-image-server-pub.git /opt/lingva-server
   cd /opt/lingva-server
   sudo -u lingva python3 -m venv .venv
   sudo -u lingva .venv/bin/pip install -r requirements.txt
   ```

   torch 不写死 CUDA 版本：从 https://pytorch.org/get-started/locally/ 按服务器情况选择安装命令。
   只有 CPU 的服务器使用 CPU 版本，例如：

   ```bash
   sudo -u lingva .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

2. 配置 `.env` 并下载模型

   ```bash
   sudo -u lingva cp .env.example .env
   sudo -u lingva nano .env   # 设置 USE_STUB=0 和随机 API_TOKEN
   sudo chmod 600 .env
   sudo -u lingva .venv/bin/python scripts/download_models.py
   ```

3. systemd

   ```bash
   sudo cp deploy/lingva.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now lingva
   journalctl -u lingva -f    # 查看启动日志中的设备与精度
   ```

4. Caddy（HTTPS 反向代理）

   ```bash
   sudo apt install caddy
   sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
   # 设置站点地址：有域名填域名；只有 IP 则填 IP，并取消 Caddyfile 中 `tls internal` 的注释
   sudo systemctl edit caddy     # 加入：[Service] Environment=LINGVA_SITE=ocr.example.com
   sudo systemctl restart caddy
   ```

5. 防火墙只开放 80 / 443（uvicorn 只监听 127.0.0.1:8000）

   ```bash
   sudo ufw allow OpenSSH
   sudo ufw allow 80,443/tcp
   sudo ufw enable
   ```

6. 验证：`curl https://<站点地址>/v1/health`
