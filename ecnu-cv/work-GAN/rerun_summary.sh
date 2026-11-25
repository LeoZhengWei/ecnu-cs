#!/bin/bash
set -e
cd /home/hjadmin/zw/ecnu/ecnu-cv/work-GAN
PY=/home/hjadmin/anaconda3/envs/zw-cv/bin/python
$PY evaluate/summary.py --data_root data/camvid_inpaint --n_test 101 --n_val 90 \
  --epochs 40 --warmup_epochs 0 --batch_size 4 --lr 0.0002 --lr_d 0.00002 \
  --lambda_adv 0.05 --img_size 256 --hole 96 --overlap 7 --jitter 24 \
  --base_context 64 --num_workers 2 --seed 42 \
  --save_dir runs_camvid --model context --loss l2 > runs_camvid/summary.log 2>&1
echo "=== summary result ==="
cat runs_camvid/summary.log
echo "=== figures ==="
ls -la runs_camvid/figures/
echo "SUMMARY_DONE"
