#!/bin/bash
set -e
cd /home/hjadmin/zw/ecnu/ecnu-cv/work-sematic-seg
PY=/home/hjadmin/anaconda3/envs/zw-cv/bin/python
export CUDA_VISIBLE_DEVICES=1
echo "ENV $(hostname) $(date)"
$PY -c "import torch;print('torch',torch.__version__,'device',torch.cuda.get_device_name(0))"
for M in efficientnet efficientnet_pt; do
  echo "===== START $M $(date) ====="
  $PY main.py --model "$M" --epochs 40 --batch_size 8 --lr 0.0003 --save_dir runs > "runs/train_${M}.log" 2>&1
  echo "===== DONE $M $(date) ====="
  tail -16 "runs/train_${M}.log"
done
echo "===== COMPARE $(date) ====="
$PY evaluate/compare.py > runs/compare.log 2>&1
tail -20 runs/compare.log
echo "ALL_DONE $(date)"
