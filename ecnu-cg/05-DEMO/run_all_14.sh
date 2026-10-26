#!/bin/bash
# ============================================================================
#  i3dgs 演示一键脚本  ——  目标机 172.88.88.14 (4x RTX 4090)
#  用法:
#     bash run_all.sh              # 自动挑最空闲的 4090，2 倍降采样训练 + 渲染漫游
#     bash run_all.sh --ds 4       # 指定 4 倍降采样（更快，画质略低）
#     bash run_all.sh --scene counter
#     bash run_all.sh --viewer     # 只启动交互式查看服务端
# ============================================================================
set -o pipefail

# ---------- 路径（迁移后布局） ----------
PROJ=/home/hjadmin/zhengwei/i3dgs-demo
BIG=/datafast/zhengwei-i3dgs
ENV=$BIG/envs/i3dgs
PY=$ENV/bin/python
DATA=$BIG/data/MipNeRF360          # 软链到 $PROJ/data
WEIGHTS=$BIG/weights               # DA3 权重
RESULTS=$BIG/results               # 软链到 $PROJ/results

# ---------- 编译目标：4090 = Ada = sm_89 ----------
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.9"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

SCENE=garden
DS=2
MODE=train

while [ $# -gt 0 ]; do
  case "$1" in
    --ds)      DS="$2"; shift 2 ;;
    --scene)   SCENE="$2"; shift 2 ;;
    --viewer)  MODE=viewer; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *)         echo "未知参数: $1"; exit 1 ;;
  esac
done

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---------- 0. 环境自检 ----------
log "===== 0/4 环境自检 ====="
[ -x "$PY" ] || { echo "找不到 conda 环境: $ENV"; exit 1; }
"$PY" - <<'EOF'
import torch, sys
print("torch      :", torch.__version__, "| cuda", torch.version.cuda)
print("gpu count  :", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  gpu{i}     : {torch.cuda.get_device_name(i)}")
mods = ["diff_gaussian_rasterization", "simple_knn._C", "fused_ssim", "depth_anything_3.api"]
for m in mods:
    try:
        __import__(m); print("OK   ", m)
    except Exception as e:
        print("FAIL ", m, e); sys.exit(1)
EOF
[ $? -eq 0 ] || { echo "环境不完整，请先跑 setup_env_14.sh"; exit 1; }

log "数据集: $DATA/$SCENE  ($(ls "$DATA/$SCENE/images" 2>/dev/null | wc -l) 张图)"

# ---------- 1. 挑一张最空闲的卡 ----------
log "===== 1/4 选择空闲 GPU ====="
pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | awk -F', *' '{print $1, $2}' | sort -k2 -nr | head -1
}
read -r GPU FREE <<< "$(pick_gpu)"
log "GPU$GPU 空闲 ${FREE} MiB"
if [ "$FREE" -lt 6000 ]; then
  log "ERROR: 最空闲的卡只剩 ${FREE} MiB，4 张卡都不够。请等别的任务结束，或把 --ds 调大。"
  nvidia-smi
  exit 1
fi
if [ "$FREE" -lt 10000 ] && [ "$DS" -lt 4 ]; then
  log "显存偏紧，自动改为 --ds 4"
  DS=4
fi
export CUDA_VISIBLE_DEVICES=$GPU

# ---------- 2. 建模 ----------
OUT=$RESULTS/${SCENE}_ds${DS}
rm -rf "$OUT"
mkdir -p "$OUT" "$BIG/logs"
TLOG=$BIG/logs/train_${SCENE}_ds${DS}.log

log "===== 2/4 开始重建: $SCENE  ds=$DS  -> $OUT ====="
log "日志: $TLOG"
cd "$PROJ/i3dgs" || exit 1
START=$(date +%s)
"$PY" -u train.py \
  -s "$DATA/$SCENE" \
  -m "$OUT" \
  --downsampling "$DS" \
  --da3_model_dir "$WEIGHTS/DA3-BASE" 2>&1 | tee "$TLOG"
RC=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END-START))
if [ $RC -ne 0 ]; then
  log "ERROR: 训练失败 (rc=$RC)，完整日志见 $TLOG"
  tail -30 "$TLOG"
  exit $RC
fi
log "===== 训练完成，耗时 ${ELAPSED}s ($((ELAPSED/60)) 分 $((ELAPSED%60)) 秒) ====="

# ---------- 3. 导出漫游视频 ----------
log "===== 3/4 渲染漫游视频 ====="
VDIR=$OUT/video
mkdir -p "$VDIR"
"$PY" -u "$PROJ/i3dgs/scripts/render_path.py" \
  -m "$OUT" --render_path "$OUT/sparse/0" --out_dir "$VDIR" 2>&1 | tail -20
VID=$(ls -1 "$VDIR"/*.mp4 2>/dev/null | head -1)
log "视频: ${VID:-未生成（逐帧图像在 $VDIR）}"

# ---------- 4. 汇总 ----------
log "===== 4/4 汇总 ====="
{
  echo "scene       = $SCENE"
  echo "downsampling= $DS"
  echo "gpu         = GPU$GPU ($(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU"))"
  echo "train_sec   = $ELAPSED"
  echo "model_dir   = $OUT"
  echo "video       = ${VID:-none}"
} | tee "$OUT/SUMMARY.txt"

log "交互式查看（本机连服务器时用）:"
log "  服务器: cd $PROJ/i3dgs && CUDA_VISIBLE_DEVICES=$GPU $PY gaussianviewer.py server $OUT"
log "  本地  : $PY gaussianviewer.py client <服务器IP>"
log "完成。"
