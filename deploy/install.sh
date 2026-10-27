#!/usr/bin/env bash
# LingVa 图片识别服务一键部署（Linux + Docker）。
#
# 两种方式都默认拉取预构建镜像（几分钟装好），不在本机构建：
#   拉取代码后执行：
#     git clone https://github.com/MuShanYu/lingva-image-server-pub.git && cd lingva-image-server-pub
#     sudo bash deploy/install.sh
#   或一行命令安装（不需要源码）：
#     curl -fsSL https://raw.githubusercontent.com/MuShanYu/lingva-image-server-pub/main/deploy/install.sh | sudo bash
# 改过源码、需要用本地代码构建镜像时，在源码目录中执行：sudo LINGVA_BUILD=yes bash deploy/install.sh
#
# 交互式：依次询问是否使用国内镜像、域名、是否使用 GPU。加 -y 则全部使用默认值或下面的环境变量：
#   LINGVA_DOMAIN=ocr.example.com   有域名时自动启用 HTTPS（Caddy + Let's Encrypt）；留空则用 HTTP
#   LINGVA_PORT=8000                不用域名时对外的端口
#   LINGVA_BIND=0.0.0.0             不用域名时监听的地址；只给内网用时填本机内网 IP（如前端 + GPU 机分开部署时的 GPU 机）
#   LINGVA_GPU=auto|yes|no          默认 auto：检测到 NVIDIA GPU 时使用
#   LINGVA_CN_MIRROR=yes|no         国内镜像：国内镜像仓库（阿里云 ACR）、pip 清华源、hf-mirror.com
#   LINGVA_DIR=/opt/lingva-server   一行命令安装时的安装目录
#   LINGVA_VERSION=latest           拉取预构建镜像时使用的版本（如 v1.0.0）
#   LINGVA_BUILD=yes                在源码目录中执行时用本地源码构建镜像（首次 5~15 分钟），默认拉取预构建镜像
#   GITHUB_PROXY=https://.../       GitHub 下载代理前缀（一行命令安装时下载部署文件用，可选）
#   LINGVA_ACR_VPC=yes              与 ACR 同地域的阿里云 ECS 改从 VPC 内网地址拉取镜像（默认用公网地址）
#
# 重复执行即为升级 / 修改配置：保留 .env 里已有的访问令牌和其他参数，只更新部署相关的几项。
set -euo pipefail

# ---------- 发布地址（发布前改成实际的仓库 / 镜像）----------
GITHUB_REPO="${LINGVA_GITHUB_REPO:-MuShanYu/lingva-image-server-pub}"
BRANCH="${LINGVA_BRANCH:-main}"
IMAGE_GLOBAL="${LINGVA_IMAGE_GLOBAL:-ghcr.io/mushanyu/lingva-server-pub}"
# 国内镜像仓库（阿里云 ACR）；留空则国内也用上面的。默认用公网地址，LINGVA_ACR_VPC=yes 时同地域 ECS 改走 VPC 内网地址
IMAGE_CN="${LINGVA_IMAGE_CN:-crpi-zexpf9vtgz3dg4i1.cn-shanghai.personal.cr.aliyuncs.com/shanghai-mirror/lingva-server-pub}"
# 国内用的 Caddy：release 工作流把 Docker Hub 的 caddy:2 同步到 IMAGE_CN 所在的命名空间
CADDY_CN="${LINGVA_CADDY_IMAGE_CN:-${IMAGE_CN%/*}/caddy:2}"

ASSUME_YES=0
for arg in "$@"; do
    case "$arg" in
        -y | --yes) ASSUME_YES=1 ;;
        -h | --help)
            # 打印开头的注释块（第 2 行起，到第一个非注释行为止）
            awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]:-/dev/null}" 2>/dev/null ||
                echo "用法见 README"
            exit 0
            ;;
        *)
            echo "未知参数：$arg（-h 查看用法）" >&2
            exit 1
            ;;
    esac
done

info() { printf '\033[1;34m[lingva]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[lingva]\033[0m %s\n' "$*" >&2; }
die() {
    printf '\033[1;31m[lingva]\033[0m %s\n' "$*" >&2
    exit 1
}

# ask "问题" "默认值"：非交互（-y 或没有终端）时直接用默认值。curl | bash 时标准输入是脚本本身，从 /dev/tty 读
ask() {
    local reply=""
    if [ "$ASSUME_YES" = 1 ] || ! (: </dev/tty) 2>/dev/null; then
        echo "$2"
        return
    fi
    read -r -p "$1 [$2]: " reply </dev/tty || true
    echo "${reply:-$2}"
}

yes_no() {
    case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
        y | yes | 1 | true) echo yes ;;
        *) echo no ;;
    esac
}

# acr_vpc 镜像地址：默认原样返回（公网地址）。LINGVA_ACR_VPC=yes 时，与 ACR 同地域的阿里云 ECS 改走 VPC 内网地址：
# 不占公网带宽、不收流量费，但私有仓库要用内网地址 docker login。
#   新版个人版 crpi-<实例>.<地域>.personal.cr.aliyuncs.com → crpi-<实例>-vpc.<地域>.personal.cr.aliyuncs.com
#   旧版         registry.<地域>.aliyuncs.com                → registry-vpc.<地域>.aliyuncs.com
acr_vpc() {
    local image=$1 registry vpc=""
    if [ "${ALIYUN_ECS:-no}" = yes ] && [ "$(yes_no "${LINGVA_ACR_VPC:-no}")" = yes ]; then
        registry=${image%%/*}
        case "$registry" in
            crpi-*."$ALIYUN_REGION".personal.cr.aliyuncs.com) vpc="${registry%%.*}-vpc.${registry#*.}" ;;
            registry."$ALIYUN_REGION".aliyuncs.com) vpc="registry-vpc.${registry#registry.}" ;;
        esac
    fi
    if [ -n "$vpc" ]; then echo "$vpc/${image#*/}"; else echo "$image"; fi
}

[ "$(uname -s)" = "Linux" ] || die "一键部署脚本只支持 Linux。Windows / macOS 请安装 Docker Desktop 后按 README 手动执行 docker compose。"
command -v curl >/dev/null 2>&1 || die "需要 curl，请先安装（如 apt install -y curl）"

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    command -v sudo >/dev/null 2>&1 || die "请用 root 运行，或先安装 sudo"
    SUDO="sudo"
fi

# 源码目录里执行（旁边有 Dockerfile）→ 源码模式，就地使用源码目录；curl | bash 等其他方式 → 发布模式（下载部署文件）
SCRIPT_PATH="${BASH_SOURCE[0]:-}"
MODE=release
if [ -n "$SCRIPT_PATH" ] && [ -f "$SCRIPT_PATH" ]; then
    SRC_ROOT="$(cd "$(dirname "$(readlink -f "$SCRIPT_PATH")")/.." && pwd)"
    [ -f "$SRC_ROOT/Dockerfile" ] && MODE=source
fi
# 是否拉取预构建镜像（而不是在本机构建）：默认拉取；源码模式下设 LINGVA_BUILD=yes 时用本地源码构建。
# 兼容旧变量：LINGVA_PREBUILT=no 等同于 LINGVA_BUILD=yes
PULL=yes
if [ "$MODE" = source ]; then
    BUILD_DEFAULT=no
    [ -n "${LINGVA_PREBUILT:-}" ] && [ "$(yes_no "$LINGVA_PREBUILT")" = no ] && BUILD_DEFAULT=yes
    [ "$(yes_no "${LINGVA_BUILD:-$BUILD_DEFAULT}")" = yes ] && PULL=no
fi
# 源码目录里的 docker-compose.yml 带 build 配置；拉取镜像时改用发布版的 compose 文件
COMPOSE_BASE=docker-compose.yml
[ "$MODE" = source ] && [ "$PULL" = yes ] && COMPOSE_BASE=docker-compose.release.yml

# ---------- 1. 收集配置 ----------
CN_MIRROR=$(yes_no "${LINGVA_CN_MIRROR:-$(ask '服务器在中国大陆、需要使用国内镜像吗？(y/n)' n)}")
DOMAIN=${LINGVA_DOMAIN-$(ask '域名（已解析到本机；留空则不启用 HTTPS，直接用 IP + 端口）' '')}
DOMAIN=$(echo "$DOMAIN" | sed -e 's#^https\?://##' -e 's#/.*$##')
PORT=${LINGVA_PORT:-8000}
if [ -z "$DOMAIN" ]; then
    PORT=$(ask '对外端口' "$PORT")
fi

# 阿里云 ECS：源码构建时 apt / pip / torch 改走 VPC 内网镜像（不占公网带宽，实测比公网镜像快数倍）。
# 访问 100.100.x 要绕过系统代理，否则代理会把元数据服务和内网镜像的请求也转走
ALIYUN_ECS=no
ALIYUN_REGION=$(curl -fsS --noproxy '*' -m 2 http://100.100.100.200/latest/meta-data/region-id 2>/dev/null || true)
if [ -n "$ALIYUN_REGION" ]; then
    ALIYUN_ECS=yes
    # 本脚本里的 curl / apt 也不让代理转发阿里云内网地址
    export no_proxy="${no_proxy:+$no_proxy,}.aliyuncs.com,100.100.100.200"
    export NO_PROXY="$no_proxy"
fi

HAS_GPU=no
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    HAS_GPU=yes
elif command -v lspci >/dev/null 2>&1 && lspci | grep -qi nvidia; then
    warn "检测到 NVIDIA 显卡，但没有可用的驱动（nvidia-smi 不可用），将按 CPU 部署。"
    if [ "$ALIYUN_ECS" = yes ]; then
        warn "阿里云 GPU 实例请先安装驱动：vGPU 规格（sgn/vgn 系列）执行"
        warn "  acs-plugin-manager --exec --plugin gpu_grid_driver_install"
        warn "（系统需为 Ubuntu 22.04/20.04、Alibaba Cloud Linux 等驱动支持的版本），装好后重新执行本脚本。"
    else
        warn "请先安装 NVIDIA 驱动，确认 nvidia-smi 正常后重新执行本脚本。"
    fi
fi
GPU_CHOICE=${LINGVA_GPU:-auto}
if [ "$GPU_CHOICE" = "auto" ]; then
    if [ "$HAS_GPU" = yes ]; then
        GPU_CHOICE=$(yes_no "$(ask '检测到 NVIDIA GPU，是否使用 GPU 加速？(y/n)' y)")
    else
        GPU_CHOICE=no
    fi
else
    GPU_CHOICE=$(yes_no "$GPU_CHOICE")
fi

# ---------- 2. Docker ----------
if ! command -v docker >/dev/null 2>&1; then
    [ "$(yes_no "$(ask '未检测到 Docker，是否自动安装？(y/n)' y)")" = yes ] ||
        die "请先安装 Docker：https://docs.docker.com/engine/install/"
    info "安装 Docker ..."
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    if [ "$CN_MIRROR" = yes ]; then
        $SUDO sh /tmp/get-docker.sh --mirror Aliyun
    else
        $SUDO sh /tmp/get-docker.sh
    fi
    $SUDO systemctl enable --now docker >/dev/null 2>&1 || true
fi

DOCKER="docker"
if ! docker info >/dev/null 2>&1; then
    DOCKER="$SUDO docker"
    $DOCKER info >/dev/null 2>&1 || die "无法连接 Docker 守护进程，请确认 Docker 已启动（systemctl start docker）"
fi
$DOCKER compose version >/dev/null 2>&1 || die "需要 Docker Compose v2（docker compose 命令），请升级 Docker"

# 国内拉取的镜像（服务和 Caddy）都在 ACR 上；只有本机构建时要从 Docker Hub 拉 python 基础镜像
if [ "$CN_MIRROR" = yes ] && [ "$PULL" = no ] && ! $DOCKER info 2>/dev/null | grep -qi "registry mirrors"; then
    warn "Docker 尚未配置镜像加速器，本机构建需要从 Docker Hub 拉取 python 基础镜像，国内服务器可能失败。"
    warn "如卡在拉取镜像，请在 /etc/docker/daemon.json 配置 registry-mirrors 后重启 Docker 再重新执行本脚本。"
fi

if [ "$GPU_CHOICE" = yes ] && ! $DOCKER info 2>/dev/null | grep -qi nvidia && command -v apt-get >/dev/null 2>&1 &&
    [ "$(yes_no "$(ask 'Docker 中没有 NVIDIA 运行时，是否自动安装 nvidia-container-toolkit？(y/n)' y)")" = yes ]; then
    # 国内用中科大镜像，nvidia.github.io 在国内很慢
    NCT_BASE="https://nvidia.github.io/libnvidia-container"
    [ "$CN_MIRROR" = yes ] && NCT_BASE="https://mirrors.ustc.edu.cn/libnvidia-container"
    info "安装 nvidia-container-toolkit（来源：$NCT_BASE）..."
    curl -fsSL "$NCT_BASE/gpgkey" | $SUDO gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] $NCT_BASE/stable/deb/\$(ARCH) /" |
        $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    $SUDO apt-get update -qq && $SUDO apt-get install -y -qq nvidia-container-toolkit
    $SUDO nvidia-ctk runtime configure --runtime=docker >/dev/null
    $SUDO systemctl restart docker
fi
if [ "$GPU_CHOICE" = yes ] && ! $DOCKER info 2>/dev/null | grep -qi nvidia; then
    warn "Docker 中未检测到 NVIDIA 运行时（nvidia-container-toolkit），GPU 可能无法使用。"
    warn "安装方法：https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    [ "$(yes_no "$(ask '仍然按 GPU 方式部署吗？选 n 改用 CPU (y/n)' n)")" = yes ] || GPU_CHOICE=no
fi

# ---------- 3. 准备目录 ----------
if [ "$MODE" = source ]; then
    cd "$SRC_ROOT"
    if [ "$PULL" = yes ]; then info "源码目录：$SRC_ROOT（拉取预构建镜像，不在本机构建）"; else info "源码构建模式：$SRC_ROOT"; fi
else
    INSTALL_DIR="${LINGVA_DIR:-/opt/lingva-server}"
    [ -z "$SUDO" ] || die "一行命令安装需要 root 权限：请用 curl ... | sudo bash"
    mkdir -p "$INSTALL_DIR/deploy"
    cd "$INSTALL_DIR"
    RAW_BASE="${GITHUB_PROXY:-}https://raw.githubusercontent.com/$GITHUB_REPO/$BRANCH"
    info "下载部署文件到 $INSTALL_DIR（来源：$RAW_BASE）..."
    fetch_file() {
        curl -fsSL --retry 3 --connect-timeout 15 "$RAW_BASE/$1" -o "$2.tmp" || die "下载失败：$RAW_BASE/$1"
        mv "$2.tmp" "$2"
    }
    fetch_file docker-compose.release.yml docker-compose.yml
    fetch_file docker-compose.gpu.yml docker-compose.gpu.yml
    fetch_file deploy/Caddyfile.docker deploy/Caddyfile.docker
    fetch_file .env.example .env.example
fi

# ---------- 4. 写 .env ----------
[ -f .env ] || cp .env.example .env

# set_kv KEY VALUE：有则替换，无则追加
set_kv() {
    if grep -q "^$1=" .env; then
        sed -i "s|^$1=.*|$1=$2|" .env
    else
        printf '%s=%s\n' "$1" "$2" >>.env
    fi
}
get_kv() { grep "^$1=" .env | tail -n 1 | cut -d= -f2- || true; }

TOKEN=$(get_kv API_TOKEN)
if [ -z "$TOKEN" ] || [ "$TOKEN" = "dev-token" ]; then
    TOKEN=$(od -An -tx1 -N24 /dev/urandom | tr -d ' \n')
    set_kv API_TOKEN "$TOKEN"
fi
set_kv USE_STUB 0
set_kv MODEL_DIR /models

if [ -n "$DOMAIN" ]; then
    set_kv LINGVA_SITE "$DOMAIN"
    set_kv LINGVA_BIND 127.0.0.1
    set_kv COMPOSE_PROFILES https
else
    set_kv LINGVA_SITE ""
    set_kv LINGVA_BIND "${LINGVA_BIND:-0.0.0.0}"
    set_kv COMPOSE_PROFILES ""
fi
set_kv LINGVA_PORT "$PORT"

if [ "$GPU_CHOICE" = yes ]; then
    set_kv COMPOSE_FILE "$COMPOSE_BASE:docker-compose.gpu.yml"
    set_kv DEVICE cuda
    # GPU 上并发 2：实测吞吐比并发 1 高约 45%，再往上吞吐几乎不涨、延迟明显变长；已手动改过的值保留
    case "$(get_kv MAX_CONCURRENCY)" in "" | 1) set_kv MAX_CONCURRENCY 2 ;; esac
else
    set_kv COMPOSE_FILE "$COMPOSE_BASE"
    set_kv DEVICE cpu
fi

if [ "$PULL" = no ]; then
    TORCH_FLAVOR=$([ "$GPU_CHOICE" = yes ] && echo cu124 || echo cpu)
    set_kv TORCH_INDEX_URL "https://download.pytorch.org/whl/$TORCH_FLAVOR"
    set_kv ONNXRUNTIME_GPU "$([ "$GPU_CHOICE" = yes ] && echo 1 || echo 0)"
    # 镜像站的 pytorch-wheels 是平铺目录，只能按固定版本安装；2.6.0 是 cu124 的最后一个版本
    TORCH_MIRROR=""
    # pip / torch 必须走 https：wheel 没有签名，只能靠 TLS 保证没被篡改。阿里云内网镜像只有 http
    # （它的证书不含 mirrors.cloud.aliyuncs.com），所以 ECS 上只有 apt 走内网——apt 会校验软件源的 GPG 签名。
    # pip 用腾讯云镜像：阿里云 https 镜像在 ECS 上对 pip / urllib 每个请求要约 24 秒（curl 正常），pip 安装会卡住
    if [ "$ALIYUN_ECS" = yes ]; then
        info "检测到阿里云 ECS，构建时 apt 使用阿里云内网镜像，pip 使用腾讯云 https 镜像，torch 使用阿里云 https 镜像"
        set_kv APT_MIRROR "http://mirrors.cloud.aliyuncs.com"
        set_kv PIP_INDEX_URL "https://mirrors.cloud.tencent.com/pypi/simple"
        TORCH_MIRROR="https://mirrors.aliyun.com/pytorch-wheels"
    elif [ "$CN_MIRROR" = yes ]; then
        set_kv APT_MIRROR "http://mirrors.aliyun.com"
        set_kv PIP_INDEX_URL "https://pypi.tuna.tsinghua.edu.cn/simple"
        TORCH_MIRROR="https://mirrors.aliyun.com/pytorch-wheels"
    else
        set_kv APT_MIRROR ""
        set_kv PIP_INDEX_URL "https://pypi.org/simple"
    fi
    # 早期版本在 ECS 上写入过 PIP_TRUSTED_HOST（关闭 pip 的证书校验），清掉
    if grep -q "^PIP_TRUSTED_HOST=" .env; then sed -i "/^PIP_TRUSTED_HOST=/d" .env; fi
    if [ -n "$TORCH_MIRROR" ]; then
        set_kv TORCH_FIND_LINKS "$TORCH_MIRROR/$TORCH_FLAVOR/"
        set_kv TORCH_VERSION "2.6.0+$TORCH_FLAVOR"
    else
        set_kv TORCH_FIND_LINKS ""
        set_kv TORCH_VERSION ""
    fi
else
    IMAGE_BASE="$IMAGE_GLOBAL"
    if { [ "$CN_MIRROR" = yes ] || [ "$ALIYUN_ECS" = yes ]; } && [ -n "$IMAGE_CN" ]; then
        IMAGE_BASE=$(acr_vpc "$IMAGE_CN")
        [ "$IMAGE_BASE" = "$IMAGE_CN" ] || info "同地域阿里云 ECS：从 VPC 内网地址 ${IMAGE_BASE%%/*} 拉取镜像"
    fi
    VERSION="${LINGVA_VERSION:-latest}"
    if [ "$GPU_CHOICE" = yes ]; then
        TAG=$([ "$VERSION" = latest ] && echo cuda || echo "$VERSION-cuda")
    else
        TAG="$VERSION"
    fi
    set_kv LINGVA_IMAGE "$IMAGE_BASE:$TAG"
fi
# Caddy（有域名时启用）：国内从 ACR 拉取同步过去的版本，其他情况用 Docker Hub 的 caddy:2
if { [ "$CN_MIRROR" = yes ] || [ "$ALIYUN_ECS" = yes ]; } && [ -n "$CADDY_CN" ] && [ -n "$IMAGE_CN" ]; then
    set_kv CADDY_IMAGE "$(acr_vpc "$CADDY_CN")"
else
    set_kv CADDY_IMAGE ""
fi
[ "$CN_MIRROR" = yes ] && set_kv HF_ENDPOINT "https://hf-mirror.com"
chmod 600 .env
mkdir -p models

# ---------- 5. 防火墙 ----------
if command -v ufw >/dev/null 2>&1 && $SUDO ufw status 2>/dev/null | grep -q "Status: active"; then
    if [ -n "$DOMAIN" ]; then
        $SUDO ufw allow 80/tcp >/dev/null && $SUDO ufw allow 443/tcp >/dev/null
    else
        $SUDO ufw allow "$PORT/tcp" >/dev/null
    fi
    info "已在 ufw 放行端口"
fi

# ---------- 6. 启动 ----------
if [ "$PULL" = no ]; then
    # 先用 docker pull 拉基础镜像：它会依次尝试各个镜像加速器，而 BuildKit 解析元数据时遇到一个加速器出错就直接失败
    BASE_IMAGE=$(sed -n 's/^FROM[[:space:]]\{1,\}\([^[:space:]]\{1,\}\).*/\1/p' Dockerfile | head -n 1)
    [ -z "$BASE_IMAGE" ] || $DOCKER pull "$BASE_IMAGE" >/dev/null || warn "拉取基础镜像 $BASE_IMAGE 失败，继续尝试构建"
    info "构建镜像（首次约需 5~15 分钟）..."
    $DOCKER compose build
else
    info "拉取镜像 $(get_kv LINGVA_IMAGE)（约 1~3 GB）..."
    $DOCKER compose pull
fi
info "启动服务 ..."
$DOCKER compose up -d
# 重新构建 / 拉取后旧镜像会变成无标签镜像（每个约 10 GB），不清理的话几次更新就能把系统盘写满
$DOCKER image prune -f >/dev/null || true

info "等待服务就绪（首次启动需要下载约 0.9 GB 模型，视网络情况可能需要十几分钟）..."
deadline=$(($(date +%s) + 60 * 60))
last_log=""
# 只监听某个内网 IP 时，本机检查也要连那个地址
HEALTH_HOST=$(get_kv LINGVA_BIND)
case "$HEALTH_HOST" in "" | 0.0.0.0) HEALTH_HOST=127.0.0.1 ;; esac
until curl -fsS --noproxy '*' "http://$HEALTH_HOST:$PORT/v1/health" 2>/dev/null | grep -q '"stub":false'; do
    if [ "$(date +%s)" -gt "$deadline" ]; then
        die "等待超时。请查看日志：cd $(pwd) && $DOCKER compose logs -f lingva"
    fi
    if ! $DOCKER compose ps --status running --services 2>/dev/null | grep -qx lingva; then
        $DOCKER compose logs --tail 30 lingva >&2 || true
        die "服务没有正常运行（上面是最近的日志）。修正后可直接重新执行安装命令。"
    fi
    line=$($DOCKER compose logs --tail 1 --no-log-prefix lingva 2>/dev/null | tr '\r' '\n' | tail -n 1 || true)
    if [ -n "$line" ] && [ "$line" != "$last_log" ]; then
        info "  $line"
        last_log=$line
    fi
    sleep 10
done

if [ -n "$DOMAIN" ]; then
    BASE_URL="https://$DOMAIN"
    if ! curl -fsS --max-time 20 "$BASE_URL/v1/health" >/dev/null 2>&1; then
        warn "服务已启动，但暂时无法通过 $BASE_URL 访问。"
        warn "请确认域名已解析到本机、80/443 端口已在云服务器安全组放行；Caddy 申请证书可能需要一两分钟。"
    fi
else
    IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    [ "$HEALTH_HOST" = 127.0.0.1 ] || IP=$HEALTH_HOST
    BASE_URL="http://${IP:-服务器公网IP}:$PORT"
fi

cat <<EOF

================ 部署完成 ================
在 LingVa 扩展的「设置 → 图片识别服务」中填写：
  服务地址：$BASE_URL
  访问令牌：$TOKEN
EOF
if [ -z "$DOMAIN" ]; then
    cat <<EOF

注意：
  - 上面是本机检测到的 IP，云服务器请换成公网 IP，并在安全组放行 $PORT 端口。
  - 未启用 HTTPS，令牌会以明文传输。有域名时建议重新执行安装命令并填写域名。
EOF
fi
cat <<EOF

安装目录：$(pwd)（配置在 .env，模型在 models/）
常用命令（在安装目录下执行）：
  查看日志：$DOCKER compose logs -f lingva
  重启：    $DOCKER compose restart
  停止：    $DOCKER compose down
  升级：    重新执行安装命令（会保留访问令牌）
EOF
