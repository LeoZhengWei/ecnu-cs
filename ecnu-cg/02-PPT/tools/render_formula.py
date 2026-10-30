# -*- coding: utf-8 -*-
"""把七步的公式用 matplotlib mathtext 渲染成 PNG，并替换各页底部的 ASCII 公式文本行。

显示尺寸按 96 dpi ↔ 幻灯片 1280x720 坐标 1:1 换算（dpi=192 即 2 倍图）。
"""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(BASE, "formula")
os.makedirs(OUTDIR, exist_ok=True)

matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
matplotlib.rcParams["mathtext.default"] = "it"

COLOR = "#1A2230"
PNT = 17.25          # 与原来的 fontSize 23px 等效（23/96 in = 17.25 pt）
DPI = 192            # 2 倍图

FORMULAS = {
    "05": r"$s(i,j)=\frac{d_i^{\top} d_j}{\Vert d_i\Vert\,\Vert d_j\Vert}$",
    "06": r"$K^{\prime}=\mathrm{top}\text{-}5\,\{\,|M_{ij}|\,\}$",
    "07": r"$|M_{ij}|>\tau_m=500\;\Longrightarrow\;(i,j)\in E,\qquad w_{ij}=1/|M_{ij}|$",
    "08": r"$S_{t+1}=S_t\cup\left\{\,\underset{i\notin S_t}{\mathrm{argmax}}\;\sum_{j\in S_t}|M_{ij}|\,\right\}\qquad(1)$",
    "09": r"$\min\sum_{i,j}\rho\left(\Vert u_{ij}-\pi(T_i,X_j)\Vert^2\right)+\lambda\Vert\delta\Vert^2,\qquad S=B-EC^{-1}E^{\top}$",
    "10": r"$\Delta T_{\alpha}=\exp(\alpha\cdot\log\Delta T),\qquad \alpha\propto d_{\mathrm{geo}}$",
    "11": r"$d_{\mathrm{mono}}\approx s_k\,d_{\mathrm{tri}}+o_k,\qquad \mathrm{screen}(i)<\tau_l$",
}

sizes = {}
for no, tex in FORMULAS.items():
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, tex, fontsize=PNT, color=COLOR)
    out = os.path.join(OUTDIR, "f%s.png" % no)
    fig.savefig(out, dpi=DPI, transparent=True, bbox_inches="tight",
                pad_inches=0.02, format="png")
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(out).size
    sizes[no] = (w / 2.0, h / 2.0)
    print("f%s.png  %dx%d  -> 显示 %.1f x %.1f" % (no, w, h, w / 2.0, h / 2.0))

# ---- 替换各页的公式文本行为图片 ----
TEXT_RE = re.compile(
    r'<Text style=\{\{ fontSize: 23, fontWeight: \'bold\', color: \'#1A2230\', lineHeight: 1\.4 \}\}>.*?</Text>',
    re.S,
)

for no in FORMULAS:
    p = os.path.join(BASE, "slides_v2", "%s.slide" % no)
    s = open(p, encoding="utf-8").read()
    w, h = sizes[no]
    img = ('<Image src="formula/f%s.png" style={{ width: %.1f, height: %.1f }} />'
           % (no, w, h))
    s2, n = TEXT_RE.subn(img, s)
    assert n == 1, "%s 命中 %d 处" % (no, n)
    open(p, "w", encoding="utf-8").write(s2)
    print("%s.slide  replaced, w=%.1f h=%.1f" % (no, w, h))
