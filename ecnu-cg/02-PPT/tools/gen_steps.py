# -*- coding: utf-8 -*-
"""生成 PPT 第 05–11 页：七个方法步骤，每步一页（示意图 + 公式 + 这步在做什么）。
版式：1280x720；左侧文字 600，右侧图 580x362（figs/figN.png，1120x700 等比）。
"""
import os

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "slides_v2")


def esc(s):
    return (s.replace("&", "&#38;").replace("<", "&#60;").replace(">", "&#62;")
             .replace("{", "&#123;").replace("}", "&#125;"))


STEPS = [
    dict(
        no="05", mark="①",
        title="全局检索：先不问“在哪”，只问“像不像”",
        sub="每来一个关键帧，先用一个全局描述子和全部历史帧比一次，取出 20 个“可能同处”的候选——这一步完全不看拍摄顺序。",
        rows=[("输入", "新帧图像 ＋ 已收下的历史关键帧库"),
              ("做法", "MegaLoc 输出 4096 维描述子；与库内全部描述子算余弦相似度，排序取前 20"),
              ("输出", "20 个候选帧（只是“像”，还不保证几何对得上）")],
        param="K = 20 个候选　·　num_prev_keyframes_check = 20",
        param_note="论文写 MixVPR，开源代码因许可换成 MegaLoc；两者都是“一帧 → 一个向量”",
        fig="figs/fig1_vpr.png",
        cap="图 1　检索与帧序无关：n 帧只做 n 次比对，不做 n² 次。",
        formula="s(i, j) = d_iᵀ d_j / (‖d_i‖ ‖d_j‖)　，　候选集 = top-20",
        fnote="相似度只看外观、不看时间——乱序输入在这里第一次被“原谅”；代价 ≈ 1 ms / 1000 对，是整条链里最便宜的一步。",
        foot="治的是“乱”：不再假设新帧只和上一帧有关，而是和全部历史都有关。",
    ),
    dict(
        no="06", mark="②",
        title="局部粗筛：砍掉“看着像、其实没对上”的候选",
        sub="对 20 个候选各跑一次轻量局部匹配，按能对上的关键点数排序，只把前 5 个留给精确匹配。",
        rows=[("输入", "① 给出的 20 个候选帧"),
              ("做法", "XFeat 提稀疏关键点，只比坐标一致性，得到内点对数 |M_ij|"),
              ("输出", "5 个候选（外观像 ＋ 确实对上了点）")],
        param="保留 K′ = 5 个　·　约 0.5 ms / 对",
        param_note="这一步故意用“便宜但粗糙”的特征：它的任务只是排序，不是定论",
        fig="figs/fig2_xfeat.png",
        cap="图 2　20 个候选按内点数排序，只有前 5 个进入下一级。",
        formula="K′ = top-5 { |M_ij| }　，只保留前 K′ 个候选",
        fnote="淘汰的正是纯外观检索的典型假阳性：两个角落都“长得像窗台”，但一个点都对不上。",
        foot="三级串联的意义：候选集 1000 → 20 → 5 → 1，贵的操作只作用在极小的集合上。",
    ),
    dict(
        no="07", mark="③",
        title="精确验证：通过才建边，边就是后面一切判断的依据",
        sub="对 5 个候选跑 LightGlue 精确配对，RANSAC 剔除外点，内点数超过 500 才写进协视图。",
        rows=[("输入", "② 留下的 5 个候选帧"),
              ("做法", "LightGlue 精确配对 ＋ RANSAC 剔外点；内点数 |M_ij| 即“这条边有多可信”"),
              ("输出", "加权协视图 G = (M, E, W)，边权 w_ij = 1/|M_ij|")],
        param="τ_m = 500 内点　·　min_matches_edge = 500",
        param_note="边权取内点数倒数：内点越多边越短，于是图上“最短路”= 最可靠的位姿链",
        fig="figs/fig3_lightglue.png",
        cap="图 3　绿=内点、红叉=被 RANSAC 剔除；通过的帧对才成为图上的一条边。",
        formula="|M_ij| > τ_m = 500　⇒　(i, j) ∈ E　，　w_ij = 1 / |M_ij|",
        fnote="这是全链唯一产生“共视约束”的地方——后面所有位姿都由这些点对推出来，边建错，后面全歪。",
        foot="到这一步为止，一张图还没解；但“谁和谁能一起算”已经确定下来了。",
    ),
    dict(
        no="08", mark="④",
        title="选窗：从图里挑出最该一起优化的 20 帧",
        sub="不是“最近 20 帧”，而是“和当前集合连接最强的那个”——一次加一个，贪心扩张到 20 个。",
        rows=[("输入", "③ 更新后的加权协视图"),
              ("做法", "每步选“与已选集合累计内点数最大”的节点加入，直到凑满 N"),
              ("输出", "局部 BA 的变量集合 S（这一步只决定“算谁”，不动数值）")],
        param="N = 20 帧　·　num_prev_keyframes_miniba_incr = 5",
        param_note="每个节点平均只有约 10 条边，所以一次 argmax 的开销可以忽略",
        fig="figs/fig4_window.png",
        cap="图 4　实心=已选，红圈=本轮 argmax；贪心比全局最优便宜得多。",
        formula="S_t+1 = S_t ∪ { argmax_i∉S_t  Σ_j∈S_t |M_ij| }　（式 1）",
        fnote="消融 NoKFSelection：PSNR 26.29 → 24.56 dB。选窗比匹配本身更影响最终质量——窗选错，后面怎么优化都救不回。",
        foot="前四步合起来只回答一件事：哪些帧该放到同一张桌子上一起算。",
    ),
    dict(
        no="09", mark="⑤",
        title="局部 BA：真正把位姿解出来的唯一一步",
        sub="在选出的 20 帧上解一次带鲁棒核的最小二乘，用 Schur 补把问题压到“只解相机”。",
        rows=[("输入", "S 内 20 个关键帧 ＋ 约 1 万路标点（按被观测次数采样）"),
              ("做法", "LM 迭代：残差 = 重投影误差；构造正规方程后做稠密 Schur 消元，Cholesky 解相机"),
              ("输出", "相机位姿 T_i 与三维路标点 X_j —— 重建本体")],
        param="N = 20、K ≈ 1e4　·　LM 阻尼按 mul_(1 + lm) 自适应",
        param_note="每个路标只允许被协视性选出的固定子集观测，避免观测数随帧数二次增长",
        fig="figs/fig5_ba.png",
        cap="图 5　二部图求解 + 分块矩阵：C 对角可逐点求逆，B 稀疏只解相机。",
        formula="min Σ ρ( ‖u_ij − π(T_i, X_j)‖² ) + λ‖δ‖²　，　S = B − E C⁻¹ Eᵀ",
        fnote="SLAM 通常只用 5 帧时间窗；这里用 20 帧并把核函数写到 GPU 上，比 g2o / CUDA-BA 快约 5 倍。",
        foot="前四步是“准备”，第六步是“纠错”，只有这一步在解——位姿重建真正发生在这里。",
    ),
    dict(
        no="10", mark="⑥",
        title="回环修正：判定“两簇其实重叠”，再把修正摊回全图",
        sub="不重跑全局 BA：用一次焊接窗口 BA 求出 Sim(3)，再沿主干路径按测地距离插值摊开。",
        rows=[("输入", "新边让两簇之间突然出现跨簇匹配"),
              ("判定", "加权图距 > 10（min_loop_size）且跨簇有效匹配对 > 2"),
              ("做法", "两簇各扩 N/2 帧做焊接 BA 得相对变换，深度变化估尺度 → Sim(3)；Dijkstra 找主干"),
              ("输出", "全图位姿与高斯的同一份 Sim(3) 修正（高斯整棵子树跟着出生帧走）")],
        param="min_loop_size = 10（加权图距）　·　焊接窗口两簇各扩 N/2 帧",
        param_note="论文原文写“跳数 > 5”，按开源代码默认值应为加权图距 > 10",
        fig="figs/fig6_loop.png",
        cap="图 6　左：两簇间的加权图距就是回环判据；右：修正沿主干按距离摊开。",
        formula="ΔT_α = exp( α · log ΔT )　，　α ∝ 到锚点的测地距离",
        fnote="用“图上的距离”替换“全局优化的解”：牺牲一点精度，把回环代价从“再跑一次全局 BA”降到近似线性。",
        foot="这是全文最值得借鉴的工程取舍——不追求最优解，追求“够用且算得起”。",
    ),
    dict(
        no="11", mark="⑦",
        title="落位：先把深度对齐，再按屏幕尺寸压成层级",
        sub="位姿定下来之后才投放高斯：分块估深度尺度与偏移，再把屏幕上看不见的小高斯合并掉。",
        rows=[("输入", "位姿 ＋ 三角化稀疏点 ＋ 单目深度（Depth Anything 3）"),
              ("做法", "图像切 4×4 块，每块单独拟合 s、o 再插值上采样；层级按屏幕投影尺寸合并"),
              ("输出", "每关键帧固定 30 次迭代的高斯优化；显存只留“看得见”的部分")],
        param="depth_grid = 4　·　τ_l = 0.5 px　·　可并数 > max(10%, 20 万)",
        param_note="hierarchy_merge_ratio = 0.1、hierarchy_merge_min_count = 200000",
        fig="figs/fig7_gauss.png",
        cap="图 7　左：分块深度对齐；右：叶节点按屏幕尺寸合并成父节点。",
        formula="d_mono ≈ s_k · d_tri + o_k　，　screen < τ_l 且 可并数达标 → kNN 加权平均合并",
        fnote="消融 NoLoDepAlgn：26.29 → 25.41 dB。层级只压显存、不改位姿，所以能安全放在链路末端。",
        foot="7 步走完，一帧的贡献结束；下一个关键帧进来，从第 ① 步重新开始。",
    ),
]

TPL = """<Slide style={{ width: '1280px', height: '720px', backgroundColor: '#FFFFFF', padding: '0px', flexDirection: 'column' }}>
    <Box style={{ width: '1280px', height: '8px', backgroundColor: '#1E4FA8' }} />
    <Box style={{ padding: '26px 40px 0px 40px', flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <Text style={{ fontSize: 33, fontWeight: 'bold', color: '#1E4FA8' }}>%(TITLE)s</Text>
        <Text style={{ fontSize: 15, color: '#8B97A8' }}>%(NO)s / 15</Text>
    </Box>
    <Box style={{ padding: '8px 40px 0px 40px' }}>
        <Text style={{ fontSize: 20, color: '#4A5568', lineHeight: 1.4 }}>%(SUB)s</Text>
    </Box>
    <Box style={{ padding: '14px 40px 0px 40px', flexDirection: 'row', gap: 20 }}>
        <Box style={{ width: 600, flexDirection: 'column', gap: 11 }}>
%(ROWS)s
            <Box style={{ backgroundColor: '#F0F5FC', border: '1px solid #3D7BD9', borderRadius: 6, padding: '10px 16px', flexDirection: 'column' }}>
                <Text style={{ fontSize: 18, color: '#1A2230', lineHeight: 1.45 }}><b>关键取值</b>　%(PARAM)s</Text>
                <Text style={{ fontSize: 16, color: '#4A5568', lineHeight: 1.4, marginTop: 4 }}>%(PNOTE)s</Text>
            </Box>
        </Box>
        <Box style={{ width: 580, flexDirection: 'column' }}>
            <Image src="%(FIG)s" style={{ width: 580, height: 362 }} />
            <Text style={{ fontSize: 14.5, color: '#8B97A8', marginTop: 5 }}>%(CAP)s</Text>
        </Box>
    </Box>
    <Box style={{ margin: '12px 40px 0px 40px', backgroundColor: '#F7F9FC', borderLeft: '5px solid #1E4FA8', padding: '10px 18px', flexDirection: 'column' }}>
        <Text style={{ fontSize: 23, fontWeight: 'bold', color: '#1A2230', lineHeight: 1.4 }}>%(FORMULA)s</Text>
        <Text style={{ fontSize: 16.5, color: '#4A5568', lineHeight: 1.4, marginTop: 3 }}>%(FNOTE)s</Text>
    </Box>
    <Box style={{ marginTop: 'auto', padding: '10px 40px 12px 40px' }}>
        <Text style={{ fontSize: 15, color: '#8B97A8' }}>%(FOOT)s</Text>
    </Box>
</Slide>
"""

ROW_TPL = ("            <Box style={{ backgroundColor: '#F7F9FC', border: '1px solid #D6DCE5', "
           "borderRadius: 6, padding: '9px 16px' }}>\n"
           "                <Text style={{ fontSize: 18, color: '#1A2230', lineHeight: 1.45 }}>"
           "<b>%(LABEL)s</b>　%(TEXT)s</Text>\n"
           "            </Box>")

for s in STEPS:
    rows = "\n".join(ROW_TPL % dict(LABEL=esc(l), TEXT=esc(t)) for l, t in s["rows"])
    out = TPL % dict(TITLE=esc(s["title"]), NO=s["no"], SUB=esc(s["sub"]), ROWS=rows,
                     PARAM=esc(s["param"]), PNOTE=esc(s["param_note"]),
                     FIG=s["fig"], CAP=esc(s["cap"]), FORMULA=esc(s["formula"]),
                     FNOTE=esc(s["fnote"]), FOOT=esc(s["foot"]))
    p = os.path.join(OUT, s["no"] + ".slide")
    open(p, "w", encoding="utf-8").write(out)
    print("written", p, len(out))
print("DONE")
