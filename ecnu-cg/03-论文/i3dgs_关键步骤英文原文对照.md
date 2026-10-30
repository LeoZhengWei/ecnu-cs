# i3dgs 七个关键处理步骤 —— 英文原文对照

> 论文：**Immediate 3D Gaussian Splat Reconstruction of Unordered Input with Global Consistency**
> Meuleman, Franke, Zhestiankin, Montemagni, Drettakis（Inria GraphDeco / Université Côte d'Azur）
> SIGGRAPH 2026（ACM TOG）· arXiv:2607.14481
> 课程：图形和视觉前沿技术 · 组员：郑炜 / 周子懿 · 指导教师：DeepSeek

---

## 这份文档怎么用

- **主表区**：每一步给出**论文英文原文**（逐句摘录，标明 Sec. 编号）、**中译**、**关键符号的英文写法**、**源码里对应的英文标识符**。
- **附录 A**：Algorithm 1 的完整英文伪代码（附录原文，含 Sec. 指向），这是全篇最精炼的"步骤地图"，答辩前看这一页就够。
- **附录 B**：英中术语对照表，按"课堂会怎么问"排序。
- **附录 C**：**论文英文原文与开源实现不一致的三处**——老师问"你跑的和论文一样吗"时，答案在这里。

英文原文中的数学符号在 PDF 提取后原本被 `« »` 包裹，本文档已还原为可读形式（散文里的照样还原，另给出 LaTeX 排版）。

**关于原文的处理方式**：所有 quoted block 中的英文**逐字照录**，未增删改一个词，包括被动语态与 `(作者, 年份)` 引用形式；其中的**加粗**是我为课堂快速定位所加的强调（不改变字句）。中译在忠实的前提下做了语序调整，便于口头复述。个别长段落为可读性做了切分，切分处文意连续。

| 缩写 | 英文全称 | 中文 |
|---|---|---|
| VPR | Visual Place Recognition | 视觉位置识别 |
| BA | Bundle Adjustment | 光束平差 / 捆集调整 |
| KF | Keyframe | 关键帧 |
| Sim(3) | Similarity transform group | 相似变换群（旋转+平移+尺度） |
| SE(3) | Special Euclidean group | 刚体变换群（旋转+平移） |

---

## ① 全局检索：用视觉位置识别召回候选

| 项 | 内容 |
|---|---|
| **English term** | **Visual Place Recognition (VPR)** level of **two-level matching** |
| 论文出处 | Sec. 4.1 *Efficient Two-level Matching* |
| 源码位置 | `scene/vpr_model.py`（类名 `MegaLoc`，**论文写的是 MixVPR**） |
| 作用 | 从**全部**已注册关键帧里，给新来的一帧挑出 K 个最可能拍到同一片表面的候选 |

### 英文原文

> For a new incoming (or query) keyframe $I_q$, we need to find the best candidates for matching from all previously registered keyframes. Our two-level matching approach first uses a learned visual place recognition model, MixVPR (Ali-bey et al., 2023), to retrieve the $K$ most similar keyframes for each incoming frame. Importantly, MixVPR extracts a 4096-dimensional global descriptor $d$ for each image that can be compared to previously registered frames via cosine similarity, making it efficient for very large-scale retrieval.

> Formally, the similarity between query frame $I_q$ and a registered keyframe $I_i$ is
> $$s(I_q, I_i) = d_q \cdot d_i$$
> where $d_q$ and $d_i$ are the respective normalized global descriptors. We retrieve the top-$K$ candidates as $K = \text{top-}K\, s(I_q, I_i) \mid I_i \in \mathcal{M}$, where $\mathcal{M}$ is the set of all registered keyframes, using $K = 20$.

### 中译

对新来的（查询）关键帧 $I_q$，我们需要从**所有**先前已注册的关键帧中找出最匹配的候选。我们的两级匹配方法首先使用一个习得的视觉位置识别模型 MixVPR，为每一帧输入检索出最相似的 $K$ 个关键帧。重要的是，MixVPR 为每张图像提取一个 **4096 维全局描述子** $d$，可以通过余弦相似度与已注册帧比较，因此在超大规模检索中也很高效。

形式化地，查询帧 $I_q$ 与已注册关键帧 $I_i$ 的相似度为 $s(I_q,I_i)=d_q\cdot d_i$，其中 $d_q$、$d_i$ 是各自归一化后的全局描述子。我们取前 $K$ 个候选，$\mathcal{M}$ 是所有已注册关键帧的集合，取 $K=20$。

### 关键符号的英文写法

| 符号 | 英文里怎么叫 | 含义 |
|---|---|---|
| $I_q$ | query keyframe / incoming frame | 新来的查询帧 |
| $\mathcal{M}$ | set of all registered keyframes | 已注册关键帧集合 |
| $d$ | **global descriptor**（4096-dimensional） | 全局描述子 |
| $s(\cdot,\cdot)$ | cosine similarity | 余弦相似度 |
| $K$ | number of retrieved candidates | 召回候选数，论文取 20 |

### 源码里的对应

| 文件 | 英文标识符 |
|---|---|
| `scene/vpr_model.py` | `MegaLoc`（Forward 返回 `cossim_best`） |
| `scene/scene_model_loop_closure.py:186` | `torch.topk(cossim_best, n_candidates)` |

源码英文注释原文（`scene/vpr_model.py` 版权头）：

```python
# Copyright (C) 2025 - 2026, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
```

---

## ② 粗筛：只用坐标一致性，比谁对上的点多

| 项 | 内容 |
|---|---|
| **English term** | **geometric verification** by **brute-force feature matching** with XFeat |
| 论文出处 | Sec. 4.1 *Efficient Two-level Matching* |
| 源码位置 | `scene/scene_model_loop_closure.py:203`（`self.matcher.batched_evaluate_match`） |
| 作用 | 把 20 个候选压到 5 个，再把贵操作（LightGlue）限制在极少数帧对上 |

### 英文原文

> Given this efficient preliminary step, we run a first round of geometric verification on the $K$ candidates using brute-force feature matching with XFeat (Potje et al., 2024), a fast and robust local feature extractor, retaining the $K'$ keyframes with the highest inlier count:
> $$K' = \text{top-}K' \; |I_i^{BF}| \;\big|\; I_i \in K$$
> where $I_i^{BF}$ is the set of brute-force matches between $I_q$ and $I_i$, and $K' = 5$.

> Our matching is a **speed/quality cascade**: MixVPR matches at ≈ 1ms/1000 pairs, XFeat at ≈ 0.5ms/pair, and LightGlue+RANSAC at ≈ 4ms/pair, with corresponding quality gains. Each stage filters for the next, so we can only afford the slower and more accurate methods on progressively smaller candidate sets.

### 中译

在这个高效的预筛步骤之后，我们对这 $K$ 个候选做第一轮**几何验证**，方式是 XFeat 的暴力特征匹配——一个快速且鲁棒的局部特征提取器——并保留**内点数最高**的 $K'$ 个关键帧。其中 $I_i^{BF}$ 是 $I_q$ 与 $I_i$ 之间的暴力匹配集合，取 $K'=5$。

我们的匹配是一个**速度 / 质量的级联**：MixVPR 约 1ms/1000 对，XFeat 约 0.5ms/对，LightGlue+RANSAC 约 4ms/对，代价递增但质量也随之提升。每一级都在为下一级做过滤，所以那些更慢、更准的方法只在越来越小的候选集上运行。

### 关键符号的英文写法

| 符号 | 英文里怎么叫 | 含义 |
|---|---|---|
| $I_i^{BF}$ | **brute-force matches** | XFeat 暴力匹配出的点对 |
| $K'$ | retained candidates after verification | 粗筛后保留的候选数，论文取 5 |
| inlier count | **inlier count** | 通过几何验证的点数 |

### 源码里的对应

| 文件:行 | 英文原文 |
|---|---|
| `scene_model_loop_closure.py:203` | `n_matches = self.matcher.batched_evaluate_match(candidate_feats, query_feats, candidate_best_rot)` |
| `scene_model_loop_closure.py:210` | `_, best_by_matches = torch.topk(n_matches, min(n, len(n_matches)))` |
| `scene_model_loop_closure.py:240` | `torch.topk(n_matches, min(n * 2, len(n_matches)))` ← 回环检测用 $2K'$ |

---

## ③ 精匹配与建边：够 500 个内点才算"对上了"

| 项 | 内容 |
|---|---|
| **English term** | **accurate matching** with LightGlue + **RANSAC with a fundamental matrix model**; pair **validated** |
| 论文出处 | Sec. 4.1（验证）与 Sec. 4.2 *Weighted Co-visibility Graph*（建边） |
| 源码位置 | `scene/scene_model_loop_closure.py:179` 起 → `UpdateGraph` |
| 作用 | 确定"这两帧真的拍到了同一片表面"，并据此往协视图里加边 |

### 英文原文

> We then perform accurate matching on this reduced set using LightGlue (Lindenberger et al., 2023), ensuring efficiency through mixed precision and CUDA graphs. A pair is **validated** if it yields more than $\tau_m = 500$ inliers after RANSAC with a fundamental matrix model:
> $$V = \{ I_i \in K' : |I_i^{LG}| > \tau_m \}$$
> where $I_i^{LG}$ is the set of LightGlue matches between $I_q$ and $I_i$.

> *(Sec. 4.2)* Each node in the graph corresponds to a keyframe, and an edge is created between two nodes if the pair has been validated: we add the edges between $I_q$ and $V$ into the graph $G=(\mathcal{M}, E, W)$ after each new frame is tracked. The edge weight is defined from the number of inlier matches between the two keyframes:
> $$w_{ij} = 1 / \#\text{inliers}(i, j)$$
> A higher number of inliers indicates a stronger geometric relationship, i.e., a lower weight.

### 中译

然后我们在这个缩减后的集合上用 LightGlue 做**精确匹配**，通过混合精度和 CUDA graph 保证效率。若一对关键帧在**以基础矩阵为模型的 RANSAC** 之后产生超过 $\tau_m=500$ 个内点，则判定为**验证通过**。其中 $I_i^{LG}$ 是 $I_q$ 与 $I_i$ 之间的 LightGlue 匹配集合。

（第 4.2 节）图中每个节点对应一个关键帧；若某对关键帧被验证通过，就在两节点之间连一条边：每跟踪完一个新帧，我们就在图 $G=(\mathcal{M},E,W)$ 中加入 $I_q$ 与 $V$ 之间的边。边权由两关键帧之间的内点匹配数定义：$w_{ij}=1/\#\text{inliers}(i,j)$。内点越多表示几何关系越强，即**权重越低**。

### 关键符号的英文写法

| 符号 | 英文里怎么叫 | 含义 |
|---|---|---|
| $\tau_m$ | **inlier threshold** | 判定"对上了"的内点门限，论文取 500 |
| $V$ | **validated pairs** | 通过验证的关键帧对 |
| $I_i^{LG}$ | **LightGlue matches** | LightGlue 匹配出的点对 |
| $G=(\mathcal{M},E,W)$ | **weighted covisibility graph** | 加权协视图 |
| $w_{ij}$ | **edge weight** | 边权 = 内点数的倒数 |
| fundamental matrix | **fundamental matrix model** | 基础矩阵模型（RANSAC 的几何模型） |

### 源码里的对应

| 文件:行 | 英文标识符 |
|---|---|
| `scene/scene_model_loop_closure.py:179` | `batched_evaluate_match` / `remove_outliers=True` |
| `poses/ba_problem.py:649` | `# Edge weight is 1 / n_matches, so invert to score by matches.` |

---

## ④ 选窗口：在协视图上贪心扩张

| 项 | 内容 |
|---|---|
| **English term** | **connectivity-based keyframe selection** by **greedy graph traversal** |
| 论文出处 | Sec. 4.3 *Finding Well-connected Subsets in the Graph*（论文式 (1)） |
| 源码位置 | `poses/ba_problem.py:647` `select_keyframes` |
| 作用 | 决定接下来这次 BA **优化哪些帧**——选错窗，后面怎么优化都救不回 |

### 英文原文

> We consider an active set of nodes $S_t$, initialized with the incoming frame $I_q$. In the graph, $I_q$ is connected to the validated pairs as explained above. We then start a greedy graph traversal, to find the nodes (keyframes) that are strongly connected to $I_q$. At each step, we add a node that has (locally) maximum connectivity to the active set $S_t$. To find the node to add, we visit all candidate nodes, i.e., all nodes connected to a member of $S_t$. We compute the total connectivity $c$ of each such node $I_i$ to $S_t$, where connectivity is defined as:
> $$c(I_i, S_t) = \sum_{I_j \in S_t} \#\text{inliers}(i,j)$$
> We then choose the node $I_i$ with the maximum cumulative inliers, which can be written as:
> $$\textbf{(1)}\quad S_{t+1} = S_t \cup \Big\{ \operatorname*{argmax}_{I_i \in \mathcal{M} \setminus S_t} c(I_i, S_t) \Big\}$$
> We continue with such steps until we have the desired number of keyframes $N$. **Given that each graph node has a small number of connections (around 10 in our experiments), this greedy approach is very efficient.**

### 中译

我们维护一个活跃节点集 $S_t$，以新来的帧 $I_q$ 初始化。在图中，$I_q$ 与验证通过的关键帧对相连。随后我们开始一次**贪心图遍历**，找出与 $I_q$ 强连通的节点（关键帧）。每一步加入一个对活跃集 $S_t$ 具有（局部）最大连通度的节点。为找到该节点，我们考察所有候选节点，即所有与 $S_t$ 中某成员相连的节点，并计算每个候选节点 $I_i$ 到 $S_t$ 的总连通度 $c$，定义为对 $S_t$ 中所有帧的内点数求和。然后选取**累计内点数最大**的节点。不断重复，直到凑够所需的关键帧数 $N$。**由于图中每个节点的连接数很小（我们实验中约 10），这种贪心方法非常高效。**

> **答辩话术**：这一步是全文最容易被忽略、但消融掉它掉得最狠的一环——论文 Table 4 中 `NoKFSelection` 一项 26.29 → 24.56 dB。

### 关键符号的英文写法

| 符号 | 英文里怎么叫 | 含义 |
|---|---|---|
| $S_t$ | **active set** of nodes | 当前已选中的节点集 |
| $c(I_i, S_t)$ | **total connectivity** / cumulative inliers | 候选节点对活跃集的连通度 |
| $N$ | desired number of keyframes | 目标关键帧数，局部 BA 取 20 |
| greedy traversal | **greedy graph traversal** | 贪心图遍历 |

### 源码里的对应（`poses/ba_problem.py:647-670`，注释为英文原文）

```python
@torch.no_grad()
def select_keyframes(self, seed_keyframes_ids, get_n_keyframes=-1, shuffle_ratio=0.0):
    ...
    selected_ids = set(seed_keyframes_ids)
    while len(selected_ids) < get_n_keyframes:
        # Count connections to selected set for each candidate
        neighbor_counts = Counter()
        for sid in selected_ids:
            for neighbor in self.pose_graph.neighbors(sid):
                if neighbor not in selected_ids and neighbor < n_keyframes:
                    # Edge weight is 1 / n_matches, so invert to score by matches.
                    neighbor_counts[neighbor] += 1.0 / self.pose_graph.edge_weight(sid, neighbor)
        if not neighbor_counts: break
        best_candidate, best_count = neighbor_counts.most_common(1)[0]
```

> **注意**：论文式 (1) 是 $\operatorname{argmax} \sum \#\text{inliers}$，代码因为边权存的是 $1/n\_matches$，所以累加时取倒数还原，等价于原来的 argmax。**两者一致**，只是实现上的倒数换算。

---

## ⑤ 局部 BA：把位姿和路标一起优化好

| 项 | 内容 |
|---|---|
| **English term** | **local bundle adjustment**（GPU-optimized），含 **multinomial sampling** 与 **dense Schur complement** |
| 论文出处 | Sec. 5 *Efficient Local Pose and Gaussian Reconstruction*、Sec. 5.1、*Solver efficiency* |
| 源码位置 | `poses/ba_problem.py:496-509` / `poses/mini_ba.py:285` `_run_lm_iterations` |
| 作用 | 修正上端传来的粗糙位姿，同时优化三维点位置 |

### 英文原文

> *(Sec. 5)* In the following, we use the terms **observations** for the 2D screen locations of matched features and **landmarks** for the 3D points of matched features. When reconstruction starts, we bootstrap our model by exhaustively matching the first 8 well-connected keyframes, rectifying outliers using Lambda Twist (Persson and Nordberg, 2018) on the GPU as a minimal P3P solver for RANSAC. We then use Depth Anything 3 (Lin et al., 2025a) to initialize poses, focal length and landmarks, fixing inaccuracies in them with local bundle adjustment (BA).

> *(Sec. 5.1)* At each incoming frame, we perform local BA. We first use the approach described above (Sec. 4.3) to select $N=20$ keyframes, often far back in time in the set of registered keyframes. This is a higher number of keyframes than previous methods that either consider direct matches only (ORB-SLAM) or a temporal window (5 frames for DROID-SLAM), since **the out-of-order problem is harder**; the challenge is to be robust while maintaining efficiency.

> To allow efficient GPU BA, we need to use a fixed set of landmarks. For this we perform **multinomial sampling** based on the number of observations per landmark to randomly select the $K$ landmarks from those that are observed by at least two of the selected keyframes ($K=10000$ in our tests). We follow standard practice to fix $n_f$ keyframes that are used in the loss but not optimized (we use $n_f=4$).

> *(Solver efficiency)* Previous fast BA solutions (Meuleman et al., 2025) are efficient and accurate for very small problems ($N=5$), but **do not scale for the $N=20$ keyframe case** we have. We address this limitation while maintaining the GPU efficiency by allowing each landmark to be observed by a **fixed subset of keyframes** in the local window, selected based on covisibility. Furthermore, we implement GPU kernels for residual and Jacobian computation, and maintain a **dense Schur complement formulation**, which shows better performance than sparse solvers in our context. We achieve **5× speedup** against existing CPU and GPU optimizers (Kümmerle et al., 2011; Fixstars, 2024; Meuleman et al., 2025).

### 中译

（第 5 节）下文中，我们用 **observations**（观测）指匹配特征点的二维屏幕坐标，**landmarks**（路标）指匹配特征点对应的三维点。重建启动时，我们对最先的 8 个强连通关键帧做穷举匹配来自举出初始模型，并用 Lambda Twist 在 GPU 上作为 RANSAC 的最小 P3P 求解器来剔除外点。随后用 **Depth Anything 3** 初始化位姿、焦距和路标，再用局部 BA 修正其中的误差。

（第 5.1 节）每来一帧就做一次局部 BA。先用 4.3 节的方法选出 $N=20$ 个关键帧，它们在时间上往往非常靠后。这比以往的方法都要多——ORB-SLAM 只考虑直接匹配，DROID-SLAM 只用一个 5 帧的时间窗——因为**无序这个问题更难**，难点在于既要鲁棒又要高效。

为了让 GPU BA 高效，我们必须使用**固定的路标集合**。为此我们按每个路标的观测数量做**多项式采样**，从至少被两个所选关键帧观测到的路标中随机选出 $K$ 个（实验中 $K=10000$）。我们还沿用惯例固定 $n_f$ 个关键帧——它们参与损失计算但不被优化（取 $n_f=4$）。

（求解器效率）以往的快速 BA 方案在极小规模（$N=5$）上高效且精确，但**无法扩展到我们的 $N=20$ 情形**。我们的解决办法是：允许每个路标只被局部窗口内**固定的一部分关键帧**观测（依协视图挑选），从而在保持 GPU 效率的同时规避这一限制。此外我们为残差和雅可比计算写了 GPU kernel，并维持**稠密 Schur 补**形式——在我们的场景下它比稀疏求解器更快。相对现有 CPU 与 GPU 优化器取得 **5 倍加速**。

### 关键符号的英文写法

| 符号 | 英文里怎么叫 | 含义 |
|---|---|---|
| observations | **observations** | 匹配特征点的 2D 屏幕位置 |
| landmarks | **landmarks** | 匹配特征点的 3D 点 |
| $N$ | keyframes in the local window | 局部窗口关键帧数，取 20 |
| $K$ | sampled landmarks | 采样路标数，取 10000 |
| $n_f$ | fixed keyframes | 参加损失但不优化的帧数，取 4 |
| Schur complement | **dense Schur complement formulation** | 稠密 Schur 补消元 |
| multinomial sampling | **multinomial sampling** | 按观测数加权的多项式采样 |

### 源码里的对应（`poses/mini_ba.py:284-299`，注释为英文原文）

```python
def _run_lm_iterations(self, poses, f, xyz, centre, uv, cam_indices,
                       fixed_masks, original_mask2, lm, eff_iters, ...):
    for iteration in range(self.iters):
        if self.use_sparse:
            # Normal equations assembled in PyTorch from the per-observation
            # Jacobians; Huber weights from each residual. total_resid =
            # Σ valid * w_huber * (|ru|+|rv|); used for the LM acceptance check.
            if self.optimize_3Dpts:
                jtj_cam, jacxr_cam, jtj_xyz, jacxr_xyz, jtj_cam_xyz, total_resid = \
                    self._assemble_jtj(...)
                # Damping
                jtj_cam.diagonal().mul_(1 + lm)
                jtj_cam.diagonal().clamp_min_(3e7 * self.ep)
                jtj_xyz.diagonal(dim1=-2, dim2=-1).mul_(1 + lm)
                jtj_xyz.diagonal(dim1=-2, dim2=-1).clamp_min_(10 * self.ep)
```

> 这里 `mul_(1 + lm)` 就是 LM（Levenberg–Marquardt）阻尼；`Huber weights` 即论文说的鲁棒核；对应的 `jtj_cam / jtj_xyz / jtj_cam_xyz` 正是 Schur 补分块里的 $B$、$C$、$E$。

---

## ⑥ 回环：分簇判定 → 焊接 BA → 主干传播

| 项 | 内容 |
|---|---|
| **English term** | **cluster-based loop closure**；**welding window** BA；**backbone propagation** / **drift correction by propagation through the graph** |
| 论文出处 | Sec. 6 *Out-of-order Global Consistency*、6.1 / 6.2 / 6.3 |
| 源码位置 | `poses/ba_problem.py:1054` `close_loop`、`:1062` `_compute_per_keyframe_transforms` |
| 作用 | 纠正累积漂移，把被回访的区域重新接回同一个坐标系 |

### 英文原文

> *(Sec. 6)* Reconstruction error accumulates over time, resulting in **drift**. Loop closure is required to correct the drift, and consolidate everything into the same space. Traditional ordered sequence loop closure uses time stamps to identify if the candidate has not been seen recently, simplifying loop closure detection; **in the out-of-order setting, this is no longer possible.**

> *(Sec. 6.1)* We first take $2K'$ best matches (Sec. 4.1), and create two separate clusters of keyframes $C_1, C_2$. Selecting $2K'$ allows clusters of size $K'$ in both clusters if evenly distributed. We initialize the two clusters with the two keyframes that are **the most distant in terms of connectivity**. We run local reconstruction (Sec. 5) on the largest of the two clusters. To determine if the clusters are not sufficiently connected, we check **the number of hops in the covisibility graph** that separates them. If the clusters are separated by **more than $\tau_{lc}=5$ hops**, and the smaller cluster has **more than $\tau_v=2$ pairs**, a loop closure occurs.

> *(Sec. 6.2)* Once a loop closure is detected, we have two clusters of keyframes that are in separate spaces, and that require poses to be reoptimized together. We use the notion of a **"welding window"** (Mur-Artal et al., 2015), but in the new context of keyframe clusters. We expand from there to select $N/2$ keyframes on **each side** of the loop closure, allowing balanced optimization with our N-frames welding BA. To constrain landmark motion overall, we **fix $n_f$ keyframes in the older set**.

> *(Sec. 6.3)* We avoid this by exploiting our covisibility graph to efficiently find "most connected" keyframes providing reliable poses. We estimate a rigid transformation that is propagated efficiently to other poses, by determining a **backbone path using Dijkstra's algorithm** in the covisibility graph. We then compute the rigid body transformation (**SE(3)**) between the initial and re-optimized pose for the anchor keyframe in the free cluster. To obtain the full similarity transformation $\mathcal{S}_\Delta \in$ **Sim(3)**, we estimate the scale change $s_\Delta$ from **landmark depth variations** and compose it with the rigid transform. Loop edges are not yet in the graph, so the backbone goes through existing connections only. We then distribute the computed Sim(3) transformation along the backbone path, **proportionally to the geodesic distance** to the optimized keyframe. We **interpolate rotations with 6D representation** (Zhou et al., 2019) and **scale using log interpolation**.

> *(Sec. 6.3 末)* After loop closure, we add the loop matches to the covisibility graph (Sec. 4.2) so that they are used in subsequent local BA.
> Each Gaussian primitive added **keeps track of the keyframe that added it**. We apply the same Sim(3) transformation as that used for the keyframe that created each Gaussian, and we update scale and rotation accordingly.

### 中译

（第 6 节）重建误差随时间累积，产生**漂移**。必须用回环闭合来纠正漂移，把一切收拢到同一个空间里。传统有序序列的回环用**时间戳**判断某候选是否很久没出现过，这让回环检测变得简单；**但在无序设定下这条路走不通了。**

（第 6.1 节）我们先取 $2K'$ 个最佳匹配，造出两个独立的关键帧簇 $C_1, C_2$。取 $2K'$ 是为了在均匀分布时两边各能凑到 $K'$ 个。我们用**连通意义上最远**的两个关键帧来初始化这两簇，并对较大的那一簇跑一次局部重建。判断两簇是否连通不足，我们看的是**协视图上把它们分开的跳数**：若相隔**超过 $\tau_{lc}=5$ 跳**，且较小的簇有**超过 $\tau_v=2$ 对**，则发生回环。

（第 6.2 节）一旦检出回环，我们就有两簇处于不同空间的关键帧，需要把它们的位姿放在一起重新优化。我们借用 **"焊接窗口"** 的概念，但换成关键帧簇的新语境。从初始簇出发，在回环的**两侧各**选出 $N/2$ 个关键帧，用 N 帧焊接 BA 做均衡优化。为整体约束路标移动，我们**固定较老那一簇中的 $n_f$ 个关键帧**。

（第 6.3 节）我们借助协视图高效找出"连通度最高"、位姿最可靠的关键帧，从而绕开传统做法中各迭代优化。我们先在自由簇的锚点关键帧上，计算其初始位姿与重优化位姿之间的刚体变换（**SE(3)**）。为得到完整的相似变换 $\mathcal{S}_\Delta\in$ **Sim(3)**，我们从**路标深度的变化**中估计尺度变化 $s_\Delta$，再与刚体变换复合。回环边此时还没进图，所以主干只经过已有的连接。接着把算出的 Sim(3) 沿主干路**按到被优化关键帧的测地距离成比例**地分配下去。我们**用 6D 表示插值旋转**，**用对数插值处理尺度**。

（第 6.3 节末）回环之后，我们把回环匹配加回协视图，供后续局部 BA 使用。每个高斯都**记着是哪个关键帧生成了它**；我们对它施加"生成它的那一帧所受的同一个 Sim(3) 变换"，并相应更新尺度和旋转。

### 关键符号的英文写法

| 符号 | 英文里怎么叫 | 含义 |
|---|---|---|
| drift | **drift** | 累积漂移 |
| $C_1, C_2$ | **clusters** of keyframes | 两个关键帧簇 |
| hops | **number of hops** in the covisibility graph | 协视图跳数 |
| $\tau_{lc}$ | **loop closure hop threshold** | 回环跳数门限，论文取 5 |
| $\tau_v$ | minimum pairs in the smaller cluster | 小簇最少对数，论文取 2 |
| welding window | **"welding window"** | 焊接窗口 |
| free / old cluster | **free cluster** / **(fixed) old cluster** | 自由簇 / 固定的老簇 |
| backbone path | **backbone path** via Dijkstra | Dijkstra 求出的主干路径 |
| anchor keyframe | **anchor keyframe** | 锚点关键帧 |
| Sim(3) | **similarity transformation** | 相似变换（含尺度） |
| geodesic distance | **geodesic distance** | 图上的测地距离 |

### 源码里的对应

| 文件:行 | 英文标识符 |
|---|---|
| `poses/ba_problem.py:1054` | `close_loop(...)` → `_prepare_loop_sides` / `_run_loop_ba` / `_compute_per_keyframe_transforms` |
| `poses/ba_problem.py:1040-1052` | `l_dR`（rotation）、`l_scales`（scale）、`l_dt`（translation） |
| `poses/ba_problem.py:1043` | `rotated_points = torch.bmm(l_dR, old_points.unsqueeze(-1)).squeeze(-1)` |
| `poses/ba_problem.py:1049` | `new_points = l_scales.unsqueeze(1) * rotated_points + l_dt` |

源码英文注释原文：

```python
# Rotate
# (N, 3, 3) @ (N, 3, 1) -> (N, 3, 1)
# Scale and Translate
# Note: l_scales needs reshape for broadcast (N, 1)
# Update in place
return gaussian_relocation_parameters
```

---

## ⑦ 建高斯：分块深度对齐 + 渐进式层级合并

| 项 | 内容 |
|---|---|
| **English term** | **tile-wise scale and offset** alignment + **progressive Gaussian hierarchy** with **adaptive merging** |
| 论文出处 | Sec. 5.2 *Gaussian Placement and Optimization*、Sec. 7 / 7.1 |
| 源码位置 | `scene/mono_depth.py:258` `align_depth_grid`、`scene/hierarchy_cut.py:90` `check_and_create_hierarchy` |
| 作用 | 把预测深度贴到真实表面上（决定高斯往哪放），并让场景大到一定程度时不爆显存 |

### 英文原文

> *(Sec. 5.2)* Recent methods place Gaussian primitives using learned depth predictors with a **scale and offset factor**, typically computed once for the entire image (Kerbl et al., 2024). This, however, is often inaccurate (Fink et al., 2025). To overcome this limitation, **we split the images into tiles, and compute a scale and offset for each tile.** We then upsample with linear interpolation to match the input resolution. This provides much finer correction to the predicted depth, improving the quality of the positions chosen for the Gaussians.

> Using this depth, we follow Meuleman et al. (2025) in spawning new Gaussians. These Gaussians are **directly linked to the keyframe**, allowing later loop closure events (Sec. 6.1) to update the radiance field. For fast optimization, the full Gaussian model is trained for 30 iterations using a custom differentiable renderer based on Mallick et al. (2024), sampling nearby views.

> *(Sec. 7)* Processing large scenes for 3DGS requires some form of hierarchy. Such solutions require a substantial computational overhead to create or update the hierarchy that **we cannot afford**. We propose a hierarchical structure that is **fast to build and update, based on primitive screen size**. A node holds Gaussian parameters (position / opacity / rotation / scale / SH coefficients), and **parent/children IDs**. At a given point in time, small footprint leaf Gaussian primitives will be **offloaded to the CPU RAM**, freeing up VRAM for the optimization.

> *(Sec. 7.1)* We analyse the hierarchy every time a new keyframe is added, computing $S_i$, **the screen size** of each Gaussian $\mathcal{G}_i$, from the current viewpoint. We merge small, subpixel Gaussians with $S_i < \tau_l = 0.5$ pixels, and **only trigger merging when more than 20% of the Gaussians fit this criterion** to avoid too frequent merging. Within the set of primitives selected for merging, we get the **4 closest Gaussians using a fast KNN search** (Papantonakis et al., 2024) and remove duplicates to prevent a Gaussian primitive from having multiple parents. Merging is performed by **groups of 4 Gaussians using KL-divergence based merging** (Kerbl et al., 2024).
>
> When loop closure occurs we move all Gaussians in the hierarchy following the transform of the keyframe most used to spawn its children; this is done **asynchronously to avoid affecting latency**. When navigating, if a merged node contributes more than $\tau_h = 1$, we reload the finer Gaussians and update their coarse parent nodes. We use **double buffering to avoid stutter**.

### 中译

（第 5.2 节）近年的方法用一个习得的深度预测器加上**一个缩放因子和一个偏移量**来放置高斯，且通常**对整张图只算一次**。但这往往不够准。为此我们**把图像切成小块，逐块各算一组缩放与偏移**，再用线性插值上采样回输入分辨率。这给预测深度提供了精细得多的修正，改善了高斯被放置的位置质量。

拿到深度后，我们沿用 Meuleman 等人的做法生成新的高斯。这些高斯**直接绑定到关键帧上**，好让后续的回环事件能够更新整个辐射场。为快速优化，整个高斯模型训练 30 次迭代，使用一个基于 Mallick 等人的可微渲染器，并采样邻近视角。

（第 7 节）处理大场景必须引入某种层级结构。但现有结构在创建或更新时的计算开销**我们负担不起**。我们提出的层级结构**构建和更新都很快，依据是基元的屏幕尺寸**。每个节点持有高斯参数（位置 / 不透明度 / 旋转 / 尺度 / SH 系数）以及**父子 ID**。在某个时刻，占用小的叶子高斯会被**卸载到 CPU 内存**，为优化腾出显存。

（第 7.1 节）每加入一个新关键帧，我们就分析一次层级，从当前视点算出每个高斯 $\mathcal{G}_i$ 的**屏幕尺寸** $S_i$。我们合并 $S_i<\tau_l=0.5$ 像素的亚像素高斯，且**只有当超过 20% 的高斯满足该条件时才触发合并**，以免合并过于频繁。在待合并的基元集合中，我们用**快速 KNN 检索取最近的 4 个高斯**，并去重以防某个高斯有多个父节点。合并以**每 4 个高斯为一组，用 KL 散度合并**的方式完成。

回环发生时，我们按"生成其子节点的那个用得最多的关键帧"的变换来移动层级中的所有高斯；这一步**异步进行以免影响延迟**。漫游时，若某个合并节点的贡献超过 $\tau_h=1$，我们就重新载入更精细的高斯并更新其粗糙父节点。我们用**双缓冲避免卡顿**。

### 关键符号的英文写法

| 符号 | 英文里怎么叫 | 含义 |
|---|---|---|
| tile | **tile** | 图像分块（注意：论文未给数值，$4\times4$ 网格来自源码 `depth_grid_size`） |
| $S_i$ | **screen size** of a Gaussian | 高斯的屏幕投影尺寸（像素） |
| $\tau_l$ | **subpixel threshold** | 亚像素门限，论文取 0.5 px |
| $\tau_h$ | to reload finer Gaussians | 精细化重载门限，论文取 1 |
| offload | **offloaded to the CPU RAM** | 卸载到主机内存 |
| double buffering | **double buffering** | 双缓冲，避免漫游卡顿 |

### 源码里的对应

| 文件:行 | 英文标识符 |
|---|---|
| `scene/mono_depth.py:258` | `align_depth_grid(mono_idepth, kpts, width, height, sampled_mono_idepth, tri_idepth, grid_size, global_scale, global_offset, ...)` |
| `scene/mono_depth.py:276/282` | `# Fill from coarser blocks` / `# Fill from neighbour blocks` |
| `scene/hierarchy_cut.py:90` | `if mask.sum() > max(len(self.xyz) * self.hierarchy_merge_ratio, self.hierarchy_merge_min_count):` |
| `scene/hierarchy_cut.py:27-31` | `hierarchy_screen_size_threshold` / `hierarchy_merge_ratio` / `hierarchy_merge_min_count` |

---

# 附录 A　Algorithm 1 Main Pipeline（英文原文全录）

> 来源：论文附录 Algorithm 1。原表右列的 `# Sec. x` 注释一并保留，它把每一步直接指回正文。
> 这是答辩前最值得看一眼的东西——**整个方法就这 25 行**。

```
Algorithm 1  Main Pipeline
Input :  Stream of unordered RGB images I_1, I_2, ...
Output:  3DGS reconstruction

 1  M ← ∅                                   // Registered keyframes
 2  G ← (∅, ∅, ∅)                           // Covisibility graph (M, E, W)
 3  G ← ∅                                   // Active Gaussian primitives
 4  H ← ∅                                   // Gaussian hierarchy
 5  Shelf ← ∅                               // Shelved unmatched frames
 6  Bootstrap first 8 well-connected keyframes;                        // Sec. 5
 7  foreach incoming image I_q do
 8      V, K′ ← TwoLevelMatching(I_q, M);                              // Sec. 4.1
 9      if V = ∅ then
10          Shelf ← Shelf ∪ I_q;
11          continue;                       // Shelve for later (Sec. 5.3)
12      end if
13      UpdateGraph(G, I_q, V);                                        // Sec. 4.2
14      loop, C_1, C_2 ← LoopDetection(I_q, K′, G);                    // Sec. 6.1
15      S ← SelectKeyframes(I_q, G, N = 20);                           // Sec. 4.3
16      LocalBA(S);                                                    // Sec. 5.1
17      if loop then
18          WeldingBA(C_1, C_2, G);                                    // Sec. 6.2
19          BackbonePropagation(G, C_1, C_2, G);                       // Sec. 6.3
20      end if
21      PlaceGaussians(I_q, G);                                        // Sec. 5.2
22      M ← M ∪ I_q;
23      UpdateHierarchy(G, H, I_q);                                    // Sec. 7.1
24      OptimizeGaussians(M, G);                                       // Sec. 5.2
25      RetryShelved(Shelf, M, G);                                     // Sec. 5.3
26  end foreach
```

**对着上面地图说一遍流程（30 秒版本）**：

> For each incoming image, we first run **two-level matching**; if nothing validates we **shelve** it for later. Otherwise we update the **covisibility graph**, then attempt **cluster-based loop detection**. We select the most connected **20 keyframes** and run **local BA**. If a loop was detected, we run **welding BA** over both clusters and propagate the Sim(3) correction along a **Dijkstra backbone**. We then **place new Gaussians**, register the frame, **update the hierarchy**, **optimize Gaussians**, and finally **retry shelved frames**.

---

# 附录 B　英中术语对照

| English | 中文 | 出现在 |
|---|---|---|
| immediate feedback / online reconstruction | 即时反馈 / 在线重建 | 摘要、Sec. 1 |
| out-of-order / unordered input | 乱序输入 | 标题、Sec. 1 |
| global consistency | 全局一致性 | 标题、Sec. 6 |
| two-level matching | 两级匹配 | Sec. 4.1 |
| visual place recognition (VPR) | 视觉位置识别 | Sec. 4.1 |
| global descriptor | 全局描述子 | Sec. 4.1 |
| cosine similarity | 余弦相似度 | Sec. 4.1 |
| geometric verification | 几何验证 | Sec. 4.1 |
| brute-force matching | 暴力匹配 | Sec. 4.1 |
| inlier count / inliers | 内点数 / 内点 | Sec. 4.1 |
| fundamental matrix | 基础矩阵 | Sec. 4.1 |
| validated pair | 验证通过的帧对 | Sec. 4.1 / 4.2 |
| (weighted) covisibility graph | （加权）协视图 | Sec. 4.2 |
| edge weight | 边权 | Sec. 4.2 |
| greedy graph traversal | 贪心图遍历 | Sec. 4.3 |
| active set | 活跃集 | Sec. 4.3 |
| bundle adjustment (BA) | 光束平差 | Sec. 5.1 |
| observations / landmarks | 观测 / 路标 | Sec. 5 |
| bootstrap | 自举初始化 | Sec. 5 |
| multinomial sampling | 多项式采样 | Sec. 5.1 |
| Schur complement | Schur 补 | Sec. 5.1 |
| Levenberg–Marquardt damping | LM 阻尼 | 源码 `mini_ba.py` |
| Huber weights | Huber 鲁棒核权重 | 源码 `mini_ba.py` |
| drift | 漂移 | Sec. 6 |
| loop closure | 回环闭合 | Sec. 6 |
| cluster (of keyframes) | 关键帧簇 | Sec. 6.1 |
| welding window | 焊接窗口 | Sec. 6.2 |
| free / old (fixed) cluster | 自由簇 / 老簇（被固定） | Sec. 6.2 |
| backbone path | 主干路径 | Sec. 6.3 |
| anchor keyframe | 锚点关键帧 | Sec. 6.3 |
| geodesic distance | 测地距离 | Sec. 6.3 |
| similarity transform, Sim(3) | 相似变换 | Sec. 6.3 |
| 6D rotation representation | 旋转的 6D 表示 | Sec. 6.3 |
| log interpolation (of scale) | 尺度的对数插值 | Sec. 6.3 |
| tile-wise scale and offset | 分块的缩放与偏移 | Sec. 5.2 |
| screen size (of a Gaussian) | （高斯的）屏幕尺寸 | Sec. 7.1 |
| subpixel Gaussian | 亚像素高斯 | Sec. 7.1 |
| KNN search | K 近邻检索 | Sec. 7.1 |
| KL-divergence based merging | 基于 KL 散度的合并 | Sec. 7.1 |
| offload to CPU RAM | 卸载到主机内存 | Sec. 7 |
| double buffering | 双缓冲 | Sec. 7.1 |
| shelved frames | 被搁置的帧 | Sec. 5.3 |

---

# 附录 C　论文英文原文 vs 开源实现：三处不一致

> **这是答辩高风险区。** 如果在台上被问"你跑的和论文写的一样吗"，照下面答，说明我们是**照代码核对过**的，而不是照抄论文。

### C.1 视觉位置识别模型：论文 MixVPR，代码 MegaLoc

- **论文原文**（Sec. 4.1）：*"...first uses a learned visual place recognition model, **MixVPR** (Ali-bey et al., 2023)..."*
- **代码实现**：`scene/vpr_model.py` 里的类是 **`MegaLoc`**，同样输出 4096 维全局描述子、同样用余弦相似度。
- **原因**：MixVPR 的许可不允许随此代码分发，作者改用功能等价的 MegaLoc 替代。
- **答法**：> For licensing reasons, the released code replaces MixVPR with MegaLoc, which produces the same 4096-D global descriptor and is used identically.

### C.2 回环触发：论文"超过 5 跳"，代码"加权图距大于 10"

- **论文原文**（Sec. 6.1）：*"If the clusters are separated by **more than $\tau_{lc}=5$ hops**, and the smaller cluster has more than $\tau_v=2$ pairs, a loop closure occurs."*
- **代码实现**：`poses/ba_problem.py:1054` 附近用 `min_loop_size = 10`，比较的是**加权图距**（边权为内点倒数），不是跳数。
- **答法**：> The paper describes unweighted hop count with threshold 5; the released implementation compares a weighted graph distance against `min_loop_size=10`. Same intent — "are these two clusters far apart in the graph" — different metric.

### C.3 层级合并：论文"超过 20% 触发 + KL 合并"，代码是双门限 + kNN 加权平均

- **论文原文**（Sec. 7.1）：*"...**only trigger merging when more than 20%** of the Gaussians fit this criterion... Merging is performed by groups of 4 Gaussians using **KL-divergence based merging**."*
- **代码实现**（`scene/hierarchy_cut.py:90`）：
  ```python
  if mask.sum() > max(len(self.xyz) * self.hierarchy_merge_ratio,
                      self.hierarchy_merge_min_count):
  ```
  其中 `hierarchy_merge_ratio = 0.1`（**不是 20%**）、`hierarchy_merge_min_count = 200000`；合并方式在 `scene/hierarchy.py` 附近为 **kNN 分组后按 `opacity · screen_size²` 加权平均**，并非 KL 投影。
- **答法**：> The released code uses a dual threshold — 10% ratio OR 200k absolute count, whichever is larger — instead of the paper's single 20% rule, and averaging instead of KL projection. The 200k floor means merging stays off for small scenes entirely.

---

## 资料来源与行号定位

| 来源 | 位置 |
|---|---|
| 论文原文 PDF | `论文/i3dgs_SIGGRAPH2026_原文.pdf`（14.8 MB，著作权归原作者与 ACM） |
| 英文原文文本 | `论文/_src_clean.txt`（arXiv HTML 全文提取，本文引文据此校对） |
| 中文全译 | `论文/i3dgs_论文中文全译.md`（约 2.77 万字） |
| 开源代码 | `172.88.88.12:/home/hjadmin/zw/ecnu/ecnv-cg/i3dgs` |
| 典型片段 | `scene/vpr_model.py`、`scene/scene_model_loop_closure.py:179-240`、`poses/ba_problem.py:496-509 / 647-672 / 1040-1062`、`poses/mini_ba.py:275-300`、`scene/mono_depth.py:258-282`、`scene/hierarchy_cut.py:90` |

> 英文原文段落保留论文**逐字**表述，包括被动语态与 (作者, 年份) 引用形式；中译在忠实的前提下做了语序调整，便于课堂口头复述。
