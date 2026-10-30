# -*- coding: utf-8 -*-
"""在 i3dgs/train.py 的 update_pbar 开头插入一行逐帧时间戳打印，可回滚。

用法：python _patch_timing.py apply|restore
"""
import os
import shutil
import sys

TRAIN = "/home/hjadmin/zhengwei/i3dgs-demo/i3dgs/train.py"
BAK = TRAIN + ".timing_bak"
MARK = '        print(f"@@F {frameID} {time.time():.3f}", flush=True)'
ANCHOR = "    def update_pbar(self, pbar, metrics, runtimes, shelf_dicts, kf_stats, frameID):\n"

mode = sys.argv[1] if len(sys.argv) > 1 else "apply"
src = open(TRAIN, "r", encoding="utf-8").read()

if mode == "apply":
    if MARK in src:
        print("already patched")
        sys.exit(0)
    assert src.count(ANCHOR) == 1, "anchor not unique: %d" % src.count(ANCHOR)
    if not os.path.exists(BAK):
        shutil.copy2(TRAIN, BAK)
        print("backup ->", BAK)
    out = src.replace(ANCHOR, ANCHOR + MARK + "\n", 1)
    open(TRAIN, "w", encoding="utf-8").write(out)
    print("patched, lines:", out.count("\n"))
elif mode == "restore":
    if os.path.exists(BAK):
        shutil.copy2(BAK, TRAIN)
        os.remove(BAK)
        print("restored from backup")
    else:
        out = src.replace(MARK + "\n", "")
        open(TRAIN, "w", encoding="utf-8").write(out)
        print("stripped marker (no backup found)")
else:
    print("unknown mode")
