# -*- coding: utf-8 -*-
"""生成七个方法步骤的示意图（学术风，主蓝 #1E4FA8）。
画布 640x400 数据坐标，figsize 5.6x3.5 @200dpi = 1120x700 px，PPT 里按 560x350 插入。
所有数学符号走 mathtext（STIX），避免中文字体缺字形。
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Polygon, Rectangle)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "figs")
os.makedirs(OUT, exist_ok=True)

BLUE = "#1E4FA8"
LBLUE = "#F0F5FC"
MBLUE = "#3D7BD9"
DARK = "#1A2230"
GREY = "#4A5568"
LGREY = "#D6DCE5"
FAINT = "#EDEFF3"
RED = "#D9534F"
GREEN = "#2E8B6F"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"


def new_ax():
    fig, ax = plt.subplots(figsize=(5.6, 3.5), dpi=200)
    ax.set_xlim(0, 640)
    ax.set_ylim(0, 400)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def rbox(ax, x, y, w, h, fc="white", ec=LGREY, lw=1.0, r=6, z=2, alpha=1.0, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
                       alpha=alpha, linestyle=ls)
    ax.add_patch(p)
    return p


def arrow(ax, p1, p2, color=BLUE, lw=1.4, ls="-", z=3, style="-|>", ms=9):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms,
                        color=color, linewidth=lw, linestyle=ls, zorder=z,
                        shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def txt(ax, x, y, s, size=10, color=DARK, weight="normal", ha="left",
        va="center", z=6, rot=0):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=z, rotation=rot)


def card(ax, y=16, h=72, x=20, w=600):
    """底部公式卡"""
    rbox(ax, x, y, w, h, fc=LBLUE, ec=MBLUE, lw=1.0)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    print("saved", os.path.basename(p))


# ---------------------------------------------------------------- ① 全局检索
def fig1():
    fig, ax = new_ax()

    # 新帧
    rbox(ax, 24, 196, 106, 92, fc=LBLUE, ec=BLUE, lw=1.6)
    txt(ax, 77, 258, "新帧", size=10, weight="bold", color=BLUE, ha="center")
    txt(ax, 77, 240, "$I_q$", size=10.5, color=BLUE, ha="center", weight="bold")
    txt(ax, 77, 218, "还没有位姿", size=7.8, color=GREY, ha="center")

    # 历史帧网格 4x3
    gx, gtop, cell, gap = 196, 374, 42, 12
    hit = {1, 5, 6}
    centers = {}
    for i in range(12):
        r, c = divmod(i, 4)
        x = gx + c * (cell + gap)
        y = gtop - r * (cell + gap) - cell
        on = i in hit
        rbox(ax, x, y, cell, cell, fc=LBLUE if on else FAINT,
             ec=BLUE if on else LGREY, lw=1.7 if on else 1.0, r=4)
        centers[i] = (x, y)
        if not on:
            ax.plot([x + 9, x + cell - 9], [y + 9, y + cell - 9],
                    color="#CBD2DC", lw=0.8, zorder=3)
            ax.plot([x + 9, x + cell - 9], [y + cell - 9, y + 9],
                    color="#CBD2DC", lw=0.8, zorder=3)
        else:
            arrow(ax, (132, 248), (x - 1, y + cell / 2), color=MBLUE, lw=1.1,
                  ls=(0, (4, 3)), ms=8)
    txt(ax, gx, gtop + 12, "历史关键帧池：每帧一个全局描述子", size=8.4, color=GREY)

    # 右侧要点
    txt(ax, 436, 358, "取 top-$K$ = 20 个候选", size=9.2, color=BLUE, weight="bold")
    txt(ax, 436, 338, "只看向量相似度排序，", size=8.2, color=GREY)
    txt(ax, 436, 322, "不依赖拍摄时间先后。", size=8.2, color=GREY)
    txt(ax, 436, 296, "$n$ 帧 → $n$ 个描述子", size=9, color=DARK, weight="bold")
    txt(ax, 436, 278, "→ $n$ 次检索，不做 $n^2$ 次比对", size=8.2, color=GREY)
    txt(ax, 436, 250, "这是「乱序也能快」的起点", size=8.2, color=RED, weight="bold")

    txt(ax, 24, 168, "MegaLoc 全局描述子：4096 维，每帧一次前向", size=8.4,
        color=DARK, weight="bold")
    txt(ax, 24, 150, "论文写 MixVPR，开源代码因许可换成 MegaLoc；两者都是「一帧 → 一个向量」",
        size=7.4, color="#8B97A8")
    txt(ax, 24, 128, "开销与帧数线性：每帧一次前向 + 对全库做一次点积，与已收帧数成正比",
        size=7.6, color=GREY)
    txt(ax, 24, 110, "所以第 1000 帧和第 10 帧的检索耗时几乎一样——这是能「边拍边算」的前提",
        size=7.6, color=GREY)

    card(ax, y=16, h=74)
    txt(ax, 34, 74, "相似度（归一化余弦）", size=8.2, color=BLUE, weight="bold")
    txt(ax, 34, 52, r"$s(i,j)=\frac{d_i^{\top} d_j}{\|d_i\|\,\|d_j\|}$",
        size=12, color=DARK)
    txt(ax, 200, 56, "取 $s$ 最大的 20 个作为候选集", size=9.4, color=DARK)
    txt(ax, 200, 36, "n 帧的成本是 O(n) 次检索，而不是 O(n²) 次两两匹配", size=7.8,
        color=GREY)
    save(fig, "fig1_vpr.png")


# ---------------------------------------------------------------- ② 局部粗筛
def fig2():
    fig, ax = new_ax()

    rbox(ax, 24, 218, 96, 112, fc="white", ec=BLUE, lw=1.2)
    rbox(ax, 164, 218, 96, 112, fc="white", ec=BLUE, lw=1.2)
    txt(ax, 72, 344, "$I_q$", size=10, color=BLUE, weight="bold", ha="center")
    txt(ax, 212, 344, "候选 $I_c$", size=9.5, color=BLUE, weight="bold", ha="center")
    rng = np.random.default_rng(7)
    A = [(38 + rng.integers(0, 66), 230 + rng.integers(0, 86)) for _ in range(9)]
    B = [(178 + rng.integers(0, 66), 230 + rng.integers(0, 86)) for _ in range(9)]
    for i in range(9):
        ok = i < 7
        ax.plot([A[i][0], B[i][0]], [A[i][1], B[i][1]],
                color=GREEN if ok else RED, lw=1.0 if ok else 0.9, alpha=0.9,
                zorder=3, linestyle="-" if ok else (0, (2, 2)))
        for p in (A[i], B[i]):
            ax.add_patch(Circle(p, 2.6, facecolor=GREEN if ok else RED,
                                edgecolor="white", linewidth=0.5, zorder=5))
    txt(ax, 24, 200, "XFeat 稀疏关键点：只比坐标一致性", size=7.8, color=GREY)
    txt(ax, 24, 184, "绿 = 内点　红 = 外点（被 RANSAC 剔除）", size=7.8, color=GREY)

    # 柱状
    vals = np.array([0.92, 0.78, 0.71, 0.55, 0.51, 0.33, 0.29, 0.26, 0.21, 0.18,
                     0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01])
    bx, by, bw = 310, 158, 12
    for i, v in enumerate(vals):
        ax.add_patch(Rectangle((bx + i * (bw + 3), by), bw, v * 128,
                               facecolor=MBLUE if i < 5 else "#DDE3EC",
                               edgecolor="none", zorder=3))
    txt(ax, bx, by + 148, "20 个候选按 $|M_{ij}|$ 排序", size=8.6, color=DARK,
        weight="bold")
    txt(ax, bx, by + 132, "$|M_{ij}|$ 高 = 两帧确实拍到了同一片表面",
        size=7.8, color=GREY)
    ax.plot([bx - 3, bx + 5 * (bw + 3)], [by + 78, by + 78], color=RED, lw=1.0,
            linestyle=(0, (3, 2)), zorder=4)
    txt(ax, bx + 5 * (bw + 3) + 4, by + 78, "留 5 个", size=8.4, color=RED,
        weight="bold")
    txt(ax, bx, by - 12, "第 1 名", size=7.4, color=GREY)
    txt(ax, bx + 19 * (bw + 3) + bw, by - 12, "第 20 名", size=7.4, color=GREY, ha="right")
    txt(ax, bx, by - 32, "排序依据：能对上的点数，不是看起来像不像",
        size=7.4, color=GREY)
    txt(ax, bx, by - 47, "外观相似却一个点都对不上 → 在这里被剔掉",
        size=7.4, color=GREY)

    card(ax, y=16, h=74)
    txt(ax, 34, 74, "筛选规则", size=8.2, color=BLUE, weight="bold")
    txt(ax, 34, 52, r"$K'=$ top-5（按 $|M_{ij}|$ 排序）", size=11, color=DARK)
    txt(ax, 250, 56, "只把前 5 个候选交给精确匹配，其余直接丢弃", size=9.4, color=DARK)
    txt(ax, 250, 36, "把贵操作限制在极少数帧对上：约 0.5 ms / 对", size=7.8, color=GREY)
    save(fig, "fig2_xfeat.png")


# ---------------------------------------------------------------- ③ 精验建边
def fig3():
    fig, ax = new_ax()

    rbox(ax, 24, 224, 100, 108, fc="white", ec=BLUE, lw=1.2)
    rbox(ax, 164, 224, 100, 108, fc="white", ec=BLUE, lw=1.2)
    txt(ax, 74, 346, "$I_i$", size=10, color=BLUE, weight="bold", ha="center")
    txt(ax, 214, 346, "$I_j$", size=10, color=BLUE, weight="bold", ha="center")
    rng = np.random.default_rng(3)
    A = [(36 + rng.integers(0, 74), 234 + rng.integers(0, 86)) for _ in range(8)]
    B = [(176 + rng.integers(0, 74), 234 + rng.integers(0, 86)) for _ in range(8)]
    for i in range(8):
        good = i < 6
        ax.plot([A[i][0], B[i][0]], [A[i][1], B[i][1]],
                color=GREEN if good else RED, lw=1.05 if good else 0.95,
                alpha=0.9, zorder=3, linestyle="-" if good else (0, (2, 2)))
        ax.add_patch(Circle(A[i], 2.7, facecolor=GREEN if good else RED,
                            edgecolor="white", linewidth=0.5, zorder=5))
        ax.add_patch(Circle(B[i], 2.7, facecolor=GREEN if good else RED,
                            edgecolor="white", linewidth=0.5, zorder=5))
        if not good:
            mx, my = (A[i][0] + B[i][0]) / 2, (A[i][1] + B[i][1]) / 2
            ax.plot([mx - 4, mx + 4], [my - 4, my + 4], color=RED, lw=1.3, zorder=6)
            ax.plot([mx - 4, mx + 4], [my + 4, my - 4], color=RED, lw=1.3, zorder=6)

    txt(ax, 24, 206, "LightGlue 精确配对 + RANSAC 剔除外点", size=8.4, color=GREY)
    txt(ax, 24, 190, "绿 = 内点（计入 $|M_{ij}|$）　红叉 = 外点（丢弃）", size=7.8, color=GREY)

    # 图
    txt(ax, 330, 380, "协视图 $G=(M,E,W)$：边的集合就是证据", size=8.6, color=BLUE,
        weight="bold")
    nodes = {"i": (362, 296), "j": (462, 334), "k": (452, 240), "m": (534, 290)}
    edges = [("i", "j", 1.0), ("i", "k", 0.7), ("j", "k", 0.4), ("j", "m", 0.8),
             ("k", "m", 0.5)]
    for a, b, w in edges:
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]],
                color=MBLUE, lw=1.0 + 2.2 * w, alpha=0.75, zorder=3)
    for n, (x, y) in nodes.items():
        ax.add_patch(Circle((x, y), 11, facecolor=LBLUE, edgecolor=BLUE,
                            linewidth=1.3, zorder=5))
        txt(ax, x, y, n, size=8.5, color=BLUE, weight="bold", ha="center", z=6)
    txt(ax, 330, 356, "边粗细 ∝ 内点数", size=7.6, color=GREY)

    txt(ax, 330, 202, r"$w_{ij}=1/|M_{ij}|$", size=11.5, color=DARK, weight="bold")
    txt(ax, 330, 184, "内点越多 → 边越短 → 图上「更近」", size=8, color=GREY)
    txt(ax, 330, 168, "图上的最短路 = 最可靠的位姿链", size=8, color=GREY)
    txt(ax, 330, 144, "后面三步（选窗、BA、回环）都只在这张图上做判断，", size=7.8,
        color=RED)
    txt(ax, 330, 130, "不再回头数像素——这是整条链能实时的结构性原因。", size=7.8,
        color=RED)

    card(ax, y=16, h=74)
    txt(ax, 34, 74, "建边判据", size=8.2, color=BLUE, weight="bold")
    txt(ax, 34, 52, r"$|M_{ij}|>\tau_m=500\;\Rightarrow\;(i,j)\in E$", size=11.5, color=DARK)
    txt(ax, 300, 56, "通过才允许建边，并记下这些 2D-3D 关联", size=9.4, color=DARK)
    txt(ax, 300, 36, "这是整条链里唯一产生「共视约束」的地方", size=7.8, color=GREY)
    save(fig, "fig3_lightglue.png")


# ---------------------------------------------------------------- ④ 贪心选窗
def fig4():
    fig, ax = new_ax()

    nodes = {
        1: (58, 316), 2: (142, 348), 3: (226, 312), 4: (124, 264),
        5: (58, 226), 6: (206, 234), 7: (276, 268), 8: (222, 172),
        9: (122, 152), 10: (322, 318), 11: (302, 190), 12: (46, 168),
    }
    edges = [(1, 2, .8), (2, 3, .9), (1, 4, .6), (4, 2, .5), (4, 5, .7),
             (3, 5, .7), (4, 6, .6), (3, 6, .8), (6, 7, .5), (6, 8, .6),
             (9, 4, .5), (9, 5, .4), (9, 8, .8), (8, 11, .6), (7, 11, .7),
             (10, 3, .4), (10, 7, .5), (9, 12, .4), (5, 12, .5)]
    for a, b, w in edges:
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]],
                color="#C3CFDF", lw=0.8 + 2.0 * w, zorder=2)
    S = {4, 6, 9}
    for n, (x, y) in nodes.items():
        if n in S:
            ax.add_patch(Circle((x, y), 14, facecolor=LBLUE, edgecolor="none", zorder=3))
            ax.add_patch(Circle((x, y), 10.5, facecolor=BLUE, edgecolor="white",
                                linewidth=1.2, zorder=5))
            txt(ax, x, y, str(n), size=8.5, color="white", weight="bold", ha="center", z=6)
        else:
            rbox(ax, x - 10.5, y - 10.5, 21, 21, fc="white", ec="#9FB0C6", lw=1.0, r=4, z=4)
            txt(ax, x, y, str(n), size=8.5, color=GREY, ha="center", z=6)

    ax.add_patch(Circle((226, 312), 19, facecolor="none", edgecolor=RED,
                        linewidth=1.5, linestyle=(0, (3, 2)), zorder=7))
    txt(ax, 226, 288, "argmax", size=8.2, color=RED, weight="bold", ha="center")

    txt(ax, 40, 122, "深蓝实心 = 已选 $S_t$　空心 = 候选", size=8, color=GREY)

    # 右侧
    txt(ax, 404, 366, "迭代过程", size=8.8, color=BLUE, weight="bold")
    for i, s in enumerate([r"$S_1=\{4\}$", r"$S_2=\{4,9\}$", r"$S_3=\{4,9,6\}$",
                           r"$\cdots$", r"$S_{20}$（凑满 $N=20$）"]):
        txt(ax, 404, 346 - i * 18, s, size=8.8, color=DARK if i < 3 else GREY)

    txt(ax, 404, 232, "为什么能贪心", size=8.8, color=BLUE, weight="bold")
    txt(ax, 404, 213, "每个节点平均只有约 10 条边，", size=8.1, color=GREY)
    txt(ax, 404, 198, "一步 argmax 的开销可忽略。", size=8.1, color=GREY)
    txt(ax, 404, 176, "但它决定了 BA 的变量范围：", size=8.1, color=RED)
    txt(ax, 404, 161, "窗选错，后面怎么优化都救不回。", size=8.1, color=RED)

    card(ax, y=16, h=74)
    txt(ax, 34, 74, "式 (1)　协视图贪心扩张", size=8.2, color=BLUE, weight="bold")
    txt(ax, 34, 50,
        r"$S_{t+1}=S_t\cup\{\,\mathrm{argmax}_{i\,\notin\,S_t}\;\Sigma_{j\,\in\,S_t}\,|M_{ij}|\,\}$",
        size=11, color=DARK)
    txt(ax, 356, 56, "消融 NoKFSelection：26.29 → 24.56 dB", size=8, color=RED)
    txt(ax, 356, 38, "选窗比匹配本身更影响最终质量", size=7.6, color=GREY)
    save(fig, "fig4_window.png")


# ---------------------------------------------------------------- ⑤ 局部 BA
def fig5():
    fig, ax = new_ax()

    cams = [(56, 316), (140, 316), (224, 316), (308, 316)]
    pts = [(66, 196), (118, 196), (170, 196), (222, 196), (274, 196), (318, 196)]
    for i, c in enumerate(cams):
        for j, p in enumerate(pts):
            if (i + j) % 3 == 1:
                continue
            ax.plot([c[0], p[0]], [c[1] - 16, p[1] + 9], color="#C9D5E6",
                    lw=0.75, zorder=2)
    for i, (x, y) in enumerate(cams):
        ax.add_patch(Polygon([(x - 11, y + 10), (x + 11, y + 10), (x, y - 11)],
                             closed=True, facecolor=BLUE, edgecolor="none", zorder=4))
        txt(ax, x, y + 22, rf"$T_{i+1}$", size=8.2, color=BLUE, weight="bold", ha="center")
    for (x, y) in pts:
        ax.add_patch(Circle((x, y), 7, facecolor=LBLUE, edgecolor=MBLUE,
                            linewidth=1.1, zorder=4))
    txt(ax, 40, 172, "待优化：相机位姿 $T_i$（6 DoF）＋ 三维路标点 $X_j$", size=8.2, color=GREY)
    txt(ax, 40, 154, "窗内 $N=20$ 帧、约 1 万点（按被观测次数采样）", size=8.2, color=GREY)
    txt(ax, 40, 136, "每个点只由协视性选出的固定子集观测，避免二次增长",
        size=8.2, color=RED)

    # 矩阵  H = [[B, E], [E^T, C]]
    txt(ax, 388, 372, r"正规方程 $H\delta=-g$ 的分块", size=8.4, color=BLUE,
        weight="bold")
    ax.add_patch(Rectangle((392, 300), 76, 56, facecolor=MBLUE, edgecolor="white",
                           linewidth=1.6, zorder=3))
    ax.add_patch(Rectangle((470, 300), 50, 56, facecolor="#B9CBE8", edgecolor="white",
                           linewidth=1.6, zorder=3))
    ax.add_patch(Rectangle((392, 242), 76, 56, facecolor="#B9CBE8", edgecolor="white",
                           linewidth=1.6, zorder=3))
    ax.add_patch(Rectangle((470, 242), 50, 56, facecolor="#7FA5D8", edgecolor="white",
                           linewidth=1.6, zorder=3))
    txt(ax, 430, 328, "$B$", size=11, color="white", weight="bold", ha="center")
    txt(ax, 495, 328, "$E$", size=11, color="white", weight="bold", ha="center")
    txt(ax, 430, 270, "$E^{\\top}$", size=11, color="white", weight="bold", ha="center")
    txt(ax, 495, 270, "$C$", size=11, color="white", weight="bold", ha="center")
    txt(ax, 388, 224, "$C$ 是对角块 → 可逐点求逆", size=7.6, color=GREY)
    txt(ax, 388, 208, "$B$ 稀疏 → 消元后只剩相机未知量", size=7.6, color=GREY)
    txt(ax, 388, 186, "这是能把 BA 塞进 GPU 的原因", size=7.8, color=RED, weight="bold")

    card(ax, y=12, h=96)
    txt(ax, 34, 92, "目标函数与求解", size=8.2, color=BLUE, weight="bold")
    txt(ax, 34, 70, r"$\min\,\Sigma_{ij}\;\rho(\|u_{ij}-\pi(T_i,X_j)\|^2)+\lambda\|\delta\|^2$",
        size=10.5, color=DARK)
    txt(ax, 34, 46, r"Schur 补：$S=B-EC^{-1}E^{\top}$　→　Cholesky 分解 $S$　→　两次回代得 $\delta_T,\delta_X$",
        size=9.2, color=DARK)
    txt(ax, 34, 26, r"$u$ 像素观测 · $\pi$ 投影 · $\rho$ Huber 核 · $\lambda$ LM 阻尼；比 g2o / CUDA-BA 快约 5×",
        size=7.8, color=GREY)
    save(fig, "fig5_ba.png")


# ---------------------------------------------------------------- ⑥ 回环修正
def fig6():
    fig, ax = new_ax()

    c1 = [(50, 266), (82, 290), (114, 260), (78, 234), (116, 224), (54, 218),
          (42, 244), (96, 304)]
    c2 = [(268, 266), (300, 290), (332, 260), (296, 234), (334, 222), (274, 218),
          (260, 244), (314, 304)]
    for grp in (c1, c2):
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                if np.hypot(grp[i][0] - grp[j][0], grp[i][1] - grp[j][1]) < 42:
                    ax.plot([grp[i][0], grp[j][0]], [grp[i][1], grp[j][1]],
                            color="#C3CFDF", lw=1.0, zorder=2)
        for p in grp:
            ax.add_patch(Circle(p, 7, facecolor=LBLUE, edgecolor=MBLUE,
                                linewidth=1.1, zorder=4))
    txt(ax, 78, 326, "簇 $C_1$", size=8.8, color=BLUE, weight="bold", ha="center")
    txt(ax, 296, 326, "簇 $C_2$", size=8.8, color=BLUE, weight="bold", ha="center")

    arrow(ax, (126, 260), (256, 260), color=RED, lw=1.4, ls=(0, (4, 3)),
          style="<|-|>", ms=9)
    txt(ax, 191, 276, r"$d_G(C_1,C_2)>\tau_{lc}=10$", size=9.2, color=RED,
        weight="bold", ha="center")

    # 主干路径
    txt(ax, 380, 326, "Dijkstra 主干 + 测地插值", size=8.4, color=BLUE, weight="bold")
    path = [(376, 266), (416, 290), (458, 266), (500, 290), (542, 262), (582, 284)]
    for i in range(len(path) - 1):
        ax.plot([path[i][0], path[i + 1][0]], [path[i][1], path[i + 1][1]],
                color=RED, lw=2.4, zorder=3, alpha=0.8)
    cmap = ["#1E4FA8", "#3D7BD9", "#6E9BD8", "#9DBBE4", "#C3D5EF", "#DEE8F7"]
    for (x, y), c in zip(path, cmap):
        ax.add_patch(Circle((x, y), 8, facecolor=c, edgecolor=BLUE,
                            linewidth=0.9, zorder=5))
    txt(ax, 380, 234, r"$\Delta T_\alpha=\exp(\alpha\log\Delta T)$", size=10, color=DARK)
    txt(ax, 380, 214, r"$\alpha\propto$ 到锚点的测地距离；旋转用 6D、尺度取对数",
        size=7.4, color=GREY)
    txt(ax, 380, 196, "颜色深浅 = 修正量大小", size=7.4, color=GREY)

    rbox(ax, 20, 14, 600, 112, fc=LBLUE, ec=MBLUE, lw=1.0)
    txt(ax, 34, 108, "三步走：判定 → 求解 → 传播", size=8.4, color=BLUE, weight="bold")
    txt(ax, 34, 86, "① 判定　图上加权距离 > 10 且跨簇匹配对 > 2　→　认作回环",
        size=9.2, color=DARK)
    txt(ax, 34, 64, "② 求解　两簇各扩 $N/2$ 帧拼成焊接窗口，做一次局部 BA 得相对变换，",
        size=9.2, color=DARK)
    txt(ax, 34, 46, "　　　　　再由路标深度变化估尺度　→　组合成 Sim(3)",
        size=9.2, color=DARK)
    txt(ax, 34, 26, "③ 传播　沿主干按测地距离插值，全图位姿与高斯套同一变换：不再跑第二次全局 BA",
        size=9.0, color=RED, weight="bold")
    save(fig, "fig6_loop.png")


# ---------------------------------------------------------------- ⑦ 落位与层级
def fig7():
    fig, ax = new_ax()

    txt(ax, 24, 372, "① 分块深度对齐", size=8.6, color=BLUE, weight="bold")
    x0, y0, W, H = 34, 202, 128, 124
    ax.add_patch(Rectangle((x0, y0), W, H, facecolor="white", edgecolor=LGREY, lw=1.0))
    for k in range(1, 4):
        ax.plot([x0 + k * W / 4, x0 + k * W / 4], [y0, y0 + H], color="#E6EAF1", lw=0.8)
        ax.plot([x0, x0 + W], [y0 + k * H / 4, y0 + k * H / 4], color="#E6EAF1", lw=0.8)
    rng = np.random.default_rng(11)
    for bi in range(16):
        bx = x0 + (bi % 4) * W / 4
        by = y0 + (bi // 4) * H / 4
        xs = np.sort(rng.uniform(bx + 4, bx + W / 4 - 4, 6))
        ys = by + (xs - bx) * 0.5 + rng.normal(0, 3.5, 6) + 10
        ys = np.clip(ys, by + 4, by + H / 4 - 4)
        for xx, yy in zip(xs, ys):
            ax.add_patch(Circle((xx, yy), 1.8, facecolor=MBLUE, edgecolor="none",
                                alpha=0.55, zorder=4))
        ax.plot([bx + 4, bx + W / 4 - 4],
                [by + 4 * 0.5 + 10, by + (W / 4 - 8) * 0.5 + 10],
                color=RED, lw=1.0, zorder=5)
    ax.annotate("", xy=(x0 + W + 4, y0 - 4), xytext=(x0, y0 - 4),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=0.8))
    ax.annotate("", xy=(x0 - 6, y0 + H), xytext=(x0 - 6, y0),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=0.8))
    txt(ax, x0 + W / 2, y0 - 16, "三角化逆深度", size=7.4, color=GREY, ha="center")
    txt(ax, x0 - 16, y0 + H / 2, "单目逆深度", size=7.4, color=GREY, ha="center", rot=90)
    txt(ax, 34, 162, r"$d_{\mathrm{mono}}\approx s_k\,d_{\mathrm{tri}}+o_k$", size=10.5,
        color=RED, weight="bold")
    txt(ax, 34, 142, "每块单独拟合 $s_k,o_k$，再插值上采样", size=7.8, color=DARK)
    txt(ax, 34, 126, "近景与远景的深度分布不同，", size=7.6, color=GREY)
    txt(ax, 34, 112, "一组全局参数贴不住表面。", size=7.6, color=GREY)
    txt(ax, 34, 94, "消融 NoLoDepAlgn：26.29 → 25.41 dB", size=7.6, color=RED)

    # 层级树
    txt(ax, 320, 372, "② 渐进式高斯层级", size=8.6, color=BLUE, weight="bold")
    leaves = [(330 + i * 26, 236) for i in range(7)]
    mids = [(343 + i * 52, 284) for i in range(3)]
    tops = [(369, 328), (421, 328)]
    grp = [mids[0], mids[0], mids[1], mids[1], mids[1], mids[2], mids[2]]
    for p, m in zip(leaves, grp):
        ax.plot([p[0], m[0]], [p[1], m[1]], color="#C3CFDF", lw=1.0, zorder=2)
    for m, t in zip(mids, [tops[0], tops[0], tops[1]]):
        ax.plot([m[0], t[0]], [m[1], t[1]], color="#C3CFDF", lw=1.0, zorder=2)
    for p in leaves:
        ax.add_patch(Circle(p, 5.5, facecolor=MBLUE, edgecolor="white", lw=1.0, zorder=4))
    for m in mids:
        ax.add_patch(Circle(m, 8, facecolor=BLUE, edgecolor="white", lw=1.0, zorder=4))
    for t in tops:
        ax.add_patch(Circle(t, 10, facecolor="#0E3F8C", edgecolor="white", lw=1.0, zorder=4))
    txt(ax, 330, 206, "叶 = 原始高斯（卸载到 CPU 内存）", size=7.4, color=GREY)
    txt(ax, 330, 192, "父 = kNN 4 个最近高斯的加权平均", size=7.4, color=GREY)
    txt(ax, 330, 172, r"合并：screen $<0.5$ px 且可并数 $>\max(0.1,\;2\times10^5)$",
        size=7.8, color=DARK, weight="bold")
    txt(ax, 330, 156, "回取：节点贡献 $>1$ px（双缓冲防卡顿）", size=7.8, color=DARK)
    txt(ax, 330, 138, "只压显存，不改位姿：4050 帧场景峰值 < 370 万高斯",
        size=7.4, color=GREY)
    txt(ax, 330, 124, "显存 < 12.8 GB，层级达 8 层", size=7.4, color=GREY)

    card(ax, y=16, h=64)
    txt(ax, 34, 66, "为什么这一步在最后", size=8.2, color=BLUE, weight="bold")
    txt(ax, 34, 46, "高斯必须落在「位姿已经正确」的坐标系里：先对齐深度、再投放、最后才压层级。",
        size=9.2, color=DARK)
    txt(ax, 34, 28, "层级只影响显存与渲染速度，不改任何位姿——所以它能安全地放到链路末端。",
        size=8.2, color=GREY)
    save(fig, "fig7_gauss.png")


fig1()
fig2()
fig3()
fig4()
fig5()
fig6()
fig7()
print("ALL FIGS DONE ->", OUT)
