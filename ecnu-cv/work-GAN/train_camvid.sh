#!/bin/bash
# 在 CamVid 驾驶场景上重跑四种网络。四个模型完全同设置，忠实复现原实验、只换数据：
# 40 轮、洞 96、对抗从第 0 轮就开、lambda_adv 0.05、batch 4。
set -e
cd /home/hjadmin/zw/ecnu/ecnu-cv/work-GAN
PY=/home/hjadmin/anaconda3/envs/zw-cv/bin/python
mkdir -p runs_camvid
COMMON="--data_root data/camvid_inpaint --n_test 101 --n_val 90 --epochs 40 --warmup_epochs 0 --batch_size 4 --lr 0.0002 --lr_d 0.00002 --lambda_adv 0.05 --img_size 256 --hole 96 --overlap 7 --jitter 24 --base_unet 32 --base_context 64 --num_workers 2 --seed 42 --save_dir runs_camvid"

echo "TRAIN_START $(date)"
for spec in "unet l2" "unet joint" "context l2" "context joint"; do
  set -- $spec
  echo "=== training $1 $2 ==="
  $PY main.py $COMMON --model $1 --loss $2 > "runs_camvid/train_$1_$2.log" 2>&1
  tail -2 "runs_camvid/train_$1_$2.log"
done

echo "=== summary / figures ==="
$PY evaluate/summary.py $COMMON --model context --loss l2 > runs_camvid/summary.log 2>&1
tail -20 runs_camvid/summary.log
echo "ALL_DONE $(date)"
