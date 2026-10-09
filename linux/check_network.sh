#!/usr/bin/env bash
set -u
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$PROJECT/logs"
status=0
echo '=== OS / disk / GPU / conda ==='
uname -a
df -h "$PROJECT"
free -h
command -v nvidia-smi >/dev/null && nvidia-smi
command -v conda >/dev/null && conda env list
echo '=== HTTPS reachability (200/3xx/401/403 均说明服务器可达；不等于比赛下载权限) ==='
for url in https://www.kaggle.com https://pypi.org/simple/ https://download.pytorch.org/whl/cu128/ https://conda.anaconda.org/conda-forge/linux-64/repodata.json https://huggingface.co https://github.com; do
  printf '%s ' "$url"
  curl -I -L --connect-timeout 10 --max-time 30 -sS -o /dev/null -w 'HTTP=%{http_code} time=%{time_total}s\n' "$url" || status=1
done
echo '=== Authenticated competition access ==='
TOOLS_PREFIX=${SOLAR_TOOLS_PREFIX:-"$PROJECT/.envs/kaggle-tools"}
if [[ -x "$TOOLS_PREFIX/bin/kaggle" ]]; then
  "$TOOLS_PREFIX/bin/kaggle" competitions files -c filament-segmentation-2026 || status=1
else
  echo '尚未安装 Kaggle CLI；先运行 bash linux/setup_env.sh tools，再配置 token 后复查。'
  status=1
fi
echo '401: 凭据问题；403: 入场/规则/权限或网关问题；HTTP=000: DNS/TLS/代理/网络问题。'
echo '文件列表成功后仍需实际下载验证签名下载域名；download_data.py 会执行该验证。'
exit "$status"
