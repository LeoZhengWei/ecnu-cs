"""把七张方法步骤示意图压平并放进 report/img/，然后插入报告 3.3 节的每一步之后。"""

import base64
import io
import os
import re

from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "report.html")
FIGDIR = os.path.join(BASE, "..", "i3dgs复现汇报", "figs")
OUTIMG = os.path.join(BASE, "img")
TARGET_W = 900

FIGFILES = [
    ("fig1_vpr.png", "m1.png",
     "图 1　步骤①：全局描述子检索——按余弦相似度从全部历史帧里取前 20 个候选"),
    ("fig2_xfeat.png", "m2.png",
     "图 2　步骤②：XFeat 粗筛——按坐标一致性排序，只保留内点最多的前 5 个候选"),
    ("fig3_lightglue.png", "m3.png",
     "图 3　步骤③：精验建边——内点数超过 500 才建边，边权取内点数的倒数"),
    ("fig4_window.png", "m4.png",
     "图 4　步骤④：贪心选窗——在加权协视图上按式 (1) 扩张，凑满 20 帧"),
    ("fig5_ba.png", "m5.png",
     "图 5　步骤⑤：局部 BA——相机—路标二部图与 Schur 补消元"),
    ("fig6_loop.png", "m6.png",
     "图 6　步骤⑥：簇级回环判定与沿主干路径的修正量传播"),
    ("fig7_gauss.png", "m7.png",
     "图 7　步骤⑦：分块深度对齐与按屏幕投影尺寸触发的层级合并"),
]

os.makedirs(OUTIMG, exist_ok=True)
print("prepare figures:")
for src, dst, _cap in FIGFILES:
    p = os.path.join(FIGDIR, src)
    im = Image.open(p).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    im = Image.alpha_composite(bg, im).convert("RGB")
    if im.width > TARGET_W:
        im = im.resize((TARGET_W, int(im.height * TARGET_W / im.width)), Image.LANCZOS)
    out = os.path.join(OUTIMG, dst)
    im.save(out, format="PNG", optimize=True)
    print("  %-20s -> %-8s %dx%d  %d KB" % (src, dst, im.width, im.height,
                                            os.path.getsize(out) // 1024))

html = open(SRC, encoding="utf-8").read()

# 原有的 garden 渲染图 图 1 -> 图 8（它排在 3.3 节之后）
old_cap = "图 1　garden 场景测试视角的重建结果"
assert old_cap in html, "未找到原图 1 图注"
html = html.replace(old_cap, "图 8　garden 场景测试视角的重建结果", 1)

# 定位 3.3 节
h = html.index("<h3>3.3　方法细节</h3>")
tail_at = html.index('<h2>4　实验分析</h2>')
sec = html[h:tail_at]

blocks = re.findall(r'<p class="eq">.*?</p>', sec, flags=re.S)
print("found %d eq blocks in 3.3" % len(blocks))
assert len(blocks) == 7, "3.3 节的公式段落数不是 7"

new_sec = sec
for i in range(len(blocks) - 1, -1, -1):
    blk = blocks[i]
    src, dst, cap = FIGFILES[i]
    ins = ('\n<figure style="text-align:center;margin:8pt 0 4pt 0">'
           '<img src="img/%s" style="width:100%%"/></figure>\n'
           '<p class="caption">%s</p>\n' % (dst, cap))
    pos = new_sec.rindex(blk) + len(blk)
    new_sec = new_sec[:pos] + ins + new_sec[pos:]

html = html[:h] + new_sec + html[tail_at:]
open(SRC, "w", encoding="utf-8").write(html)
print("written:", SRC, len(html), "chars")
print("figure count:", html.count("<figure"))
