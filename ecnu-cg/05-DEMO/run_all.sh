#!/bin/bash
# =============================================================
# i3dgs 一键演示 pipeline
#   用法: bash run_all.sh [场景名] [降采样倍数]
#   默认: bash run_all.sh garden 4
#   说明: 必须在 i3dgs/ 目录下运行训练 —— 代码按 "poses/ransac.cu"
#         这类相对路径读文件，换目录会报 FileNotFoundError
# =============================================================
set -e
SCENE=${1:-garden}
DS=${2:-4}

ROOT=/home/hjadmin/zw/ecnu/ecnv-cg
I3DGS=$ROOT/i3dgs
LOGS=$ROOT/logs
DATA=$ROOT/data/MipNeRF360/$SCENE
OUT=$ROOT/results/${SCENE}_ds${DS}
mkdir -p "$LOGS" "$ROOT/results"

source /home/hjadmin/anaconda3/etc/profile.d/conda.sh
conda activate i3dgs

export HF_ENDPOINT=https://hf-mirror.com          # HF 镜像（本网络无法直连）
export TORCH_CUDA_ARCH_LIST="12.0"                # RTX 5090 = sm_120，必须 12.0
export PATH=/usr/local/cuda-12.8/bin:$PATH
export PYTHONUNBUFFERED=1

cd "$I3DGS"

echo "=============================================="
echo "[Step 0] 环境自检"
echo "=============================================="
python -c "
import torch
print('torch      :', torch.__version__, '| cuda', torch.version.cuda)
print('cuda avail :', torch.cuda.is_available(), '| devices', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)
    print('  gpu%d: %s  %.1fGB total, %.1fGB free' % (i, p.name, total/2**30, free/2**30))
for m in ['diff_gaussian_rasterization','simple_knn','cupy','depth_anything_3','graphdecoviewer','lightglue']:
    try:
        __import__(m); print('  %-30s OK' % m)
    except Exception as e:
        print('  %-30s FAIL %s' % (m, e))
"

echo "=============================================="
echo "[Step 1] 数据准备 (场景: $SCENE)"
echo "=============================================="
if [ ! -d "$DATA/images" ]; then
  echo "本地无数据，开始下载 MipNeRF360（约 1.0 GB，含 garden/counter/bonsai）"
  python scripts/download_datasets.py --out_dir "$ROOT/data" --datasets MipNeRF360
else
  echo "已存在 $DATA ($(ls "$DATA/images" | wc -l) 张图)"
fi

echo "=============================================="
echo "[Step 2] 等待可用显存 (>=8GB on gpu1) 并开始重建"
echo "=============================================="
NEED=8000
while true; do
  F=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$1==1{print $2}')
  echo "  gpu1 free = ${F} MiB / need ${NEED} MiB"
  [ "${F:-0}" -ge "$NEED" ] && break
  sleep 30
done

echo "=============================================="
echo "[Step 3] 即时重建 (CUDA_VISIBLE_DEVICES=1 -> RTX 5090, downsampling=$DS)"
echo "=============================================="
export CUDA_VISIBLE_DEVICES=1
: > "$LOGS/gpu_samples.log"
( while true; do nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader >> "$LOGS/gpu_samples.log"; sleep 5; done ) &
MON=$!
T0=$(date +%s)
python train.py -s "$DATA" -m "$OUT" --downsampling "$DS" 2>&1 | tee "$LOGS/train_${SCENE}_ds${DS}.log"
T1=$(date +%s)
kill $MON 2>/dev/null || true
echo ">>> 训练耗时: $((T1-T0)) 秒"
echo ">>> 峰值显存: $(awk -F', ' '$1==1{print $2}' "$LOGS/gpu_samples.log" | sort -n | tail -1) MiB"

echo "=============================================="
echo "[Step 4] 沿优化出的相机轨迹渲染漫游视频"
echo "=============================================="
python scripts/render_path.py -m "$OUT" \
  --render_path "$OUT/sparse/0" \
  --out_dir "$OUT/video" || echo "渲染跳过（需先确认 sparse/0 存在）"

echo "=============================================="
echo "[Step 5] 结果清单"
echo "=============================================="
echo "模型目录 : $OUT"
echo "视频     : $OUT/video"
ls -lh "$OUT/video" 2>/dev/null | head
echo
echo "现场交互演示（本地机器上另开终端）:"
echo "  服务器: python gaussianviewer.py server $OUT"
echo "  本地  : python gaussianviewer.py client --ip <server-ip>"
echo "  或直接 SSH 端口转发后用客户端连接 127.0.0.1"
echo "DONE"
