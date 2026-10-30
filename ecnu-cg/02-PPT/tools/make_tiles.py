# -*- coding: utf-8 -*-
"""把 12 张快照渲染图裁成竖版并压成 jpg，供 PPT 页面使用。"""
import json
import os

from PIL import Image

SRC = r"C:/Users/13205/WorkBuddy/2026-10-16-19-11-43/i3dgs复现汇报/prog_fix"
DST = os.path.join(SRC, "tiles")
os.makedirs(DST, exist_ok=True)

frames = [36, 40, 44, 48, 52, 60, 64, 68, 72, 76, 80, 84]
kf = {36: 8, 40: 9, 44: 11, 48: 12, 52: 14, 60: 18,
      64: 22, 68: 26, 72: 32, 76: 34, 80: 35, 84: 81}

rows = []
for i, fr in enumerate(frames, 1):
    p = os.path.join(SRC, "%05d.png" % fr)
    im = Image.open(p).convert("RGB")
    w, h = im.size
    im = im.resize((360, 480), Image.LANCZOS)
    out = os.path.join(DST, "t%02d.jpg" % i)
    im.save(out, "JPEG", quality=88, optimize=True)
    rows.append({"idx": i, "frame": fr, "kf": kf[fr], "jpg": out,
                 "bytes": os.path.getsize(out)})
    print("%2d  frame=%-3d kf=%-3d %6d B  %s" % (i, fr, kf[fr], os.path.getsize(out), out))

json.dump(rows, open(os.path.join(DST, "tiles.json"), "w"), indent=2)
print("total %.2f MB" % (sum(r["bytes"] for r in rows) / 1048576.0))
