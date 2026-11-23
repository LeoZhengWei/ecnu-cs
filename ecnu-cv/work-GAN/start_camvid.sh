#!/bin/bash
cd /home/hjadmin/zw/ecnu/ecnu-cv/work-GAN
mkdir -p runs_camvid
if [ -f runs_camvid/train_camvid.log ] && grep -q 'ALL_DONE' runs_camvid/train_camvid.log; then
  echo "ALREADY_DONE - skip"
  tail -20 runs_camvid/train_camvid.log
  exit 0
fi
if pgrep -f 'train_camvid.sh' > /dev/null; then
  echo "ALREADY_RUNNING - skip"
  pgrep -af 'train_camvid.sh'
  exit 0
fi
nohup bash train_camvid.sh > runs_camvid/train_camvid.log 2>&1 < /dev/null &
PID=$!
disown "$PID" 2>/dev/null || true
echo "STARTED pid=$PID  $(date)"
sleep 5
echo "=== master log so far"
head -20 runs_camvid/train_camvid.log 2>/dev/null
