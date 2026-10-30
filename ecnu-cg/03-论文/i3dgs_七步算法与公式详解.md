# i3dgs 七步算法与公式详解

> 论文：**Immediate 3D Gaussian Splat Reconstruction of Unordered Input with Global Consistency**
> Meuleman, Franke, Zhestiankin, Montemagni, Drettakis（Inria GraphDeco / Université Côte d'Azur）
> SIGGRAPH 2026（ACM TOG）· arXiv:2607.14481
> 课程：图形和视觉前沿技术 · 组员：郑炜 / 周子懿

**这份文档对应 PPT 的哪几页**：P4「方法总览：一帧进来，走完七步」列出七步，P5–P11 一步一页，每页底部一条公式。本文档把那七条公式**补全成可推导、可实现的形式**，并逐段标注**哪一段是本文的贡献、哪一段是借来的**。

**和另外三份文档的分工**：

| 文档 | 回答什么问题 |
|---|---|
| `i3dgs_七步算法与公式详解.md`（本文） | 七步各自的**算法怎么做、公式怎么来、参数取多少、哪一段是本文贡献** |
| `i3dgs_关键步骤英文原文对照.md` | 七步的**论文英文原句**怎么写的、英文术语怎么念 |
| `i3dgs_论文精要.md` | 10 分钟过一遍这篇论文讲什么 |
| `../04-讲稿/演讲讲稿与答辩手册.md` | 每一页怎么讲、被问什么怎么答 |

---

## 0 读法约定

每个步骤统一按七段写：

1. **干什么**（一句话，和 PPT 那页标题对齐）
2. **输入 / 输出**（数据类型，不是自然语言）
3. **算法**（可照着写的伪代码，行号与论文 Algorithm 1 对得上）
4. **公式**（LaTeX 原式 + 每个符号的含义 + 为什么是这个形式）
5. **超参数**（论文值 / 开源代码默认值 / 出处，三者不一致时明确标出）
6. **实现细节**（GPU 怎么写的、哪里省了主机同步）
7. **贡献归属**（★ 本文贡献 / ☆ 借用他人 / ○ 教科书内容）+ 消融数字

标记含义：

- **★**：本文提出的东西，答辩时要能说清"为什么非这么设计不可"
- **☆**：借用已有工作，本文只是选用并集成（要能报出引用）
- **○**：教科书内容（LM、Schur 补、Huber、Dijkstra、Cholesky），不构成创新

---

## 1 记号与符号表

| 符号 | 含义 | 在论文哪里出现 |
|---|---|---|
| $I_q$ | 新来的查询关键帧（query / incoming frame） | Sec. 4.1 |
| $\mathcal{M}$ | 已注册关键帧集合 | Sec. 4.1 |
| $d_i$ | $I_i$ 的 4096 维归一化全局描述子 | Sec. 4.1 |
| $s(I_q,I_i)$ | 两帧的余弦相似度 | Sec. 4.1 式 |
| $K,\ K'$ | 召回候选数（20）、粗筛后保留数（5） | Sec. 4.1 |
| $I_i^{BF}$ | XFeat 暴力匹配点对集合 | Sec. 4.1 |
| $I_i^{LG}$ | LightGlue 精确匹配点对集合 | Sec. 4.1 |
| $\tau_m$ | 建边所需最少内点数（500） | Sec. 4.1 |
| $G=(\mathcal{M},E,W)$ | 加权协视图 | Sec. 4.2 |
| $w_{ij}$ | 边权 $=1/\#\text{inliers}(i,j)$ | Sec. 4.2 |
| $S_t$ | 选窗时的活跃节点集 | Sec. 4.3 |
| $c(I_i,S_t)$ | 候选节点对活跃集的累计连通度 | Sec. 4.3 |
| $N$ | 局部窗口关键帧数（20） | Sec. 5.1 |
| $K$（路标） | 采样路标数（10000） | Sec. 5.1 |
| $n_f$ | 参与损失但不优化的固定帧数（4） | Sec. 5.1 |
| $T_i\in SE(3)$ | 第 $i$ 帧相机位姿 | Sec. 5.1 |
| $X_j\in\mathbb{R}^3$ | 第 $j$ 个三维路标 | Sec. 5.1 |
| $u_{ij}$ | 观测（2D 屏幕坐标） | Sec. 5 |
| $\pi(\cdot)$ | 投影函数 | Sec. 5.1 |
| $\rho(\cdot)$ | 鲁棒核（Huber） | 源码 `mini_ba.py` |
| $B,C,E$ | 正规方程分块（相机-相机 / 点-点 / 相机-点） | Sec. 5.1 |
| $S$ | Schur 补矩阵 $B-EC^{-1}E^{\top}$ | Sec. 5.1 |
| $C_1,C_2$ | 回环检测出的两个关键帧簇 | Sec. 6.1 |
| $\tau_{lc},\ \tau_v$ | 回环跳数门限（5）、小簇最少对数（2） | Sec. 6.1 |
| $\mathcal{S}_\Delta$ | Sim(3) 相似变换 | Sec. 6.3 |
| $s_\Delta$ | 回环前后的尺度变化 | Sec. 6.3 |
| $S_i$ | 高斯的屏幕投影尺寸（像素） | Sec. 7.1 |
| $\tau_l$ | 亚像素门限（0.5 px） | Sec. 7.1 |
| $\tau_h$ | 精细化重载门限（1） | Sec. 7.1 |

> ⚠️ **一个符号两用**：$K$ 在 Sec. 4.1 是"召回候选数 = 20"，在 Sec. 5.1 是"采样路标数 = 10000"。论文自己就这样复用，读的时候看章节。

---

## 2 先说结论：本文到底贡献了什么

### 2.1 三条贡献（原文逐字）

> In summary, our contributions are:
> - A fast matching approach for unordered input for efficient retrieval of well-connected subsets of keyframes, allowing GPU-friendly local reconstruction.
> - Cluster-based loop detection for unordered sequences, with efficient optimization-free updates of poses and Gaussian primitives.
> - A progressive hierarchy for large scenes that allows efficient optimization and rendering accommodating limited GPU resources.

中译：

1. **面向无序输入的快速匹配方法**，能高效检索出"强连通的关键帧子集"，从而让局部重建对 GPU 友好。
2. **面向无序序列的簇级回环检测**，以及**免优化**的位姿与高斯基元更新。
3. **面向大场景的渐进式层级**，在有限显存下仍能高效优化与渲染。

结论段还有一句总纲（原文）：

> Together, these contributions yield the first complete solution that makes immediate feedback possible for high-quality radiance field capture.

即：三条加起来，才第一次让"高质量辐射场采集过程中就能即时反馈"成为完整方案。**注意它说的是 "complete solution"，不是 "first method"** ——单项技术都不新，新的是把它们拼成一个能在乱序输入下跑起来的整体。这一点答辩被追问"创新点在哪"时必须守住。

### 2.2 三条贡献分别落在哪一步

| 贡献 | 落在 | 具体是哪一行代码/哪一段算法 |
|---|---|---|
| ① 快速匹配 + 强连通子集检索 | 步骤 **①②③④**（外加 ⑤ 的"固定路标子集"） | 三级级联匹配；协视图建边；式 (1) 贪心选窗；把 BA 规模钉死在 $N{=}20,K{=}10^4$ |
| ② 簇级回环 + 免优化更新 | 步骤 **⑥** | 双簇 + 图距判据；焊接 BA 求 Sim(3)；Dijkstra 主干 + 按测地距离摊开；高斯跟随出生关键帧 |
| ③ 渐进式层级 | 步骤 **⑦** | 按屏幕尺寸 $S_i$ 判亚像素；双门限触发合并；不可见叶子卸载到 CPU |

### 2.3 全表：七步里哪一段是本文的、哪一段是借的

这张表是本文档最该背下来的一页——**它是"这篇论文不是炼丹"的直接证据，也是被问"你自己的东西在哪"时的答案**。

| 步骤 | 本文贡献（★） | 借用（☆） | 教科书（○） |
|---|---|---|---|
| ① 全局检索 | 把 VPR 用作**乱序召回**的第一级，接进在线流水线 | MixVPR / MegaLoc（Ali-bey 2023）4096 维描述子 | 余弦相似度 |
| ② 局部粗筛 | **三级速度/质量级联**（1 ms/1000 对 → 0.5 ms/对 → 4 ms/对），逐级收窄候选集 | XFeat（Potje 2024） | top-$K$ 排序 |
| ③ 精验建边 | 用**内点数倒数当边权**，把"匹配可靠度"变成"图上的距离"，供 ④⑥ 复用 | LightGlue（Lindenberger 2023）、RANSAC + 基础矩阵 | 基础矩阵约束 |
| ④ 贪心选窗 | **式 (1)**：不用时间窗，用"对已选集合的累计内点数"贪心扩张；这直接替代了 SLAM 的 5 帧时间窗 | 协视图概念来自 ORB-SLAM（Mur-Artal 2015） | 贪心算法 |
| ⑤ 局部 BA | **每个路标只被协视性选出的固定子集观测**（破解 $N{=}20$ 时观测数二次增长）；**自写 CUDA 核 + 整轮 LM 捕获成一张 CUDA Graph** | MiniBA 思路（Meuleman 2025）、Lambda Twist P3P（Persson 2018）、DA3（Lin 2025） | LM、Huber、Schur 补、Cholesky |
| ⑥ 回环修正 | **簇级回环判据**（替代时间戳）+ **焊接 BA**（把焊接窗口搬到簇语境）+ **免优化的 Sim(3) 主干传播** | 焊接窗口（Mur-Artal 2015）、SE(3)/Sim(3)、6D 旋转插值（Zhou 2019） | Dijkstra 最短路 |
| ⑦ 高斯落位 | **渐进式层级**：按屏幕尺寸判亚像素、双门限触发、父子 ID 挂载、异步搬移 | 分块 scale-offset 是本文自述的新做法但**未列入三条贡献**；高斯生成沿用 Meuleman 2025；KL 合并（Kerbl 2024）；kNN（Papantonakis 2024） | 线性插值上采样 |

**三处值得单独记住的"本文设计"**（答辩最容易被问的三个"为什么"）：

1. **为什么回环不能用时间戳？** 乱序输入下时间戳与空间位置无关，"最近"不等于"附近"。→ 用加权图距。
2. **为什么回环传播用 Sim(3) 而不是 SE(3)？** 单目重建的尺度不可观测，回环两侧尺度天然对不上；只用刚体变换数学上不闭合。尺度 $s_\Delta$ 从路标深度变化里估。
3. **为什么 20 帧能算得动，SLAM 只用 5 帧？** 不是不想用多，是旧求解器在 $N{=}20$ 时观测数随 $N$ 二次增长，稠密 Schur 补直接爆掉。本文把每个路标的观测集固定下来，规模才可控。

### 2.4 证据：消融与加速比

**表 6（MipNeRF360，有序场景）——每去掉一个模块掉多少 dB：**

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | 去掉的是什么 |
|---|---|---|---|---|
| **Ours** | **26.29** | **0.839** | **0.241** | — |
| NoVPR | 24.78 | 0.747 | 0.291 | 不做全局检索，退回局部匹配 |
| NoMatchVerif | 25.36 | 0.812 | 0.257 | 不做 XFeat 粗筛，直接用 MixVPR 的 top-$K$ |
| **NoKFSelection** | **24.56** | 0.733 | 0.303 | 选窗退回"最近 20 帧"（**掉得最狠，−1.73 dB**） |
| NoRandFixed | 25.78 | 0.816 | 0.254 | 固定最老的相机而不是随机固定 |
| NoGraphProp | 25.81 | 0.822 | 0.249 | 用 PyPose 的 PGO 代替图传播（此项主要比速度） |
| NoGSUpdate | 25.45 | 0.814 | 0.267 | 回环后不搬高斯 |
| NoLoDepAlgn | 25.41 | 0.823 | 0.249 | 不做分块深度对齐（整图一个 scale/offset） |

**表 15（20 位姿 / 10000 路标 / 10 次迭代）——BA 求解器对比：**

| 求解器 | 耗时 | 相对本文 |
|---|---|---|
| g2o（CPU） | 461.4 ms | 29.0× |
| CudaBA | 78.1 ms | 4.9× |
| On-The-Fly (MiniBA) | 79.7 ms | 5.0× |
| **Ours** | **15.9 ms** | 1× |

以上三条合起来说明一件事：**去掉任何一个"本文设计"，要么掉 0.5–1.7 dB，要么慢 5–29 倍**。调参调不出这种表。

---

## 3 主流程：Algorithm 1（25 行）与七步的对应

论文附录 Algorithm 1 是全篇最精炼的地图。右边注释里的 `# Sec. x` 是原文自带的，把每一行直接指回正文。

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

**行号 → 七步映射：**

| 行 | 七步 | 备注 |
|---|---|---|
| 6 | 0（自举） | 前 8 个强连通帧穷举匹配 + Lambda Twist P3P + DA3 初始化 |
| 8 | ①②③ | `TwoLevelMatching` 一行包了三步，顺序是 VPR → XFeat → LightGlue/RANSAC |
| 9–11 | 容错 | 一条边都没建上就"搁置"，不是丢弃（Sec. 5.3） |
| 13 | ③（落地） | 把验证通过的边写进 $G$ |
| 14 | ⑥（判定） | 注意：**检测在 BA 之前，修正（18–19）在 BA 之后** |
| 15 | ④ | 选窗 |
| 16 | ⑤ | 局部 BA |
| 18–19 | ⑥（修正） | 只有 `loop` 为真才跑，昂贵的部分因此被摊薄 |
| 21–24 | ⑦ | 落位 → 注册 → 更新层级 → 优化高斯 |
| 25 | 容错 | 每帧末尾重试搁置帧 |

**两个容易看错的地方：**

- 第 14 行 `LoopDetection` **在局部 BA 之前**。回环先用"新边"判断要不要焊接，再照常做本轮 BA；焊接用的是 BA 之后的位姿。顺序错了逻辑就不闭合。
- 第 22 行 `M ← M ∪ I_q` **在 \(I_q\) 已经参与完匹配之后**才注册。也就是说 \(I_q\) **不会和自己匹配**，也不会出现在自己的选窗候选里——这是"新帧必须和全部历史比"的实现前提。

---

## 4 ① 全局检索：用视觉位置识别召回候选

### 4.1 干什么

每来一个关键帧，先算一个全局描述子，和**全部历史帧**比一次，取最像的 20 个。这一步完全不看拍摄顺序。

### 4.2 输入 / 输出

- 输入：新帧图像 $I_q$；已注册关键帧集合 $\mathcal{M}$ 及各自描述子 $\{d_i\}$
- 输出：候选集 $K\subseteq\mathcal{M}$，$|K|=20$

### 4.3 算法

```
1: d_q ← VPR(I_q)                       # 4096 维，L2 归一化
2: for I_i in M:  s_i ← <d_q, d_i>       # 一次矩阵乘即可批量完成
3: K ← topK(s, 20)
```

### 4.4 公式

$$s(I_q,I_i)=d_q\cdot d_i,\qquad K=\operatorname*{top-}K_{I_i\in\mathcal{M}}\;s(I_q,I_i),\quad K=20$$

- $d_q,d_i$ 是**归一化后**的描述子，所以点积就是余弦相似度，不需要再除模长。
- 之所以能用"点积 + top-k"，是因为检索只要求**排序**，不要求分数有意义——这一步不做任何几何判断。

### 4.5 超参

| 参数 | 论文 | 代码 | 备注 |
|---|---|---|---|
| 描述子维度 | 4096 | 4096 | MegaLoc 与 MixVPR 同维度 |
| $K$ | 20 | `num_prev_keyframes_check = 20` | 20 是"够用且不贵"的折中 |
| 单对耗时 | ≈1 ms / 1000 对 | — | 全链最便宜的一步 |

### 4.6 实现细节

- 描述子先入库，新帧只算一次前向；比较是 $(1\times4096)\cdot(4096\times|\mathcal{M}|)$ 的矩阵乘，GPU 上可忽略。
- 这一步**没有几何**，所以必然产生大量假阳性——这正是 ②③ 存在的理由。

### 4.7 贡献归属

- ★ **本文贡献**：把 VPR 从"重定位/回环"工具改造成**在线重建的常规第一级**，且每帧都跑。这是"乱序"能力的入口。
- ☆ 借用：MixVPR（Ali-bey 2023）；**开源代码换成 MegaLoc**（许可原因，功能等价，同为 4096 维余弦检索）。
- 消融：`NoVPR` 24.78 dB（−1.51 dB）。

---

## 5 ② 局部粗筛：只比坐标一致性

### 5.1 干什么

对 20 个候选各跑一次**便宜**的局部匹配，按"能对上的点数"排序，只留 5 个。

### 5.2 输入 / 输出

- 输入：$K$ 中的 20 个候选
- 输出：$K'$，$|K'|=5$

### 5.3 算法

```
1: for I_i in K:
2:     F_q ← XFeat(I_q);  F_i ← XFeat(I_i)     # 稀疏关键点 + 描述子
3:     M_i ← 暴力最近邻匹配(F_q, F_i)
4:     n_i ← |M_i|
5: K' ← topK'(n, 5)
```

### 5.4 公式

$$K'=\operatorname*{top-}K'_{I_i\in K}\;\bigl|I_i^{BF}\bigr|,\qquad K'=5$$

- $I_i^{BF}$（brute-force matches）是 XFeat 的暴力匹配集合。**这一级不做 RANSAC**，只数点数——因为它的任务是**排序**，不是定论。
- 代价模型是三级的核心：$\underbrace{1\,\text{ms}/1000}_{\text{VPR}}\ \rightarrow\ \underbrace{0.5\,\text{ms}/\text{对}}_{\text{XFeat}}\ \rightarrow\ \underbrace{4\,\text{ms}/\text{对}}_{\text{LightGlue}}$。贵 8 倍的操作只作用在 1/4 的候选上。

### 5.5 超参

| 参数 | 论文 | 代码 | 备注 |
|---|---|---|---|
| $K'$ | 5 | `topk(n_matches, min(n, len))` | 回环检测时取 $2K'=10$ |
| 单对耗时 | ≈0.5 ms | — | 比 LightGlue 便宜 8 倍 |

### 5.6 贡献归属

- ★ **本文贡献**：这条**三级级联**本身。单拿出 XFeat 或 LightGlue 都是别人的，但"谁先跑、跑在多大的集合上"是本文的设计。
- 消融：`NoMatchVerif` 25.36 dB（−0.93 dB）——说明粗筛不只是省时间，还**挡掉了错误建边**。

---

## 6 ③ 精验建边：够 500 个内点才算"对上了"

### 6.1 干什么

对 5 个候选跑 LightGlue 精确配对，RANSAC 剔外点；内点数超过 500 才建边。**这是全链唯一产生共视约束的地方**。

### 6.2 输入 / 输出

- 输入：$K'$ 中的 5 个候选
- 输出：验证集 $V$；加权协视图 $G=(\mathcal{M},E,W)$ 的新增边

### 6.3 算法

```
1: V ← ∅
2: for I_i in K':
3:     M_i ← LightGlue(I_q, I_i)          # 混合精度 + CUDA Graph
4:     M_i ← RANSAC(M_i, 基础矩阵模型)     # 剔外点
5:     if |M_i| > τ_m:
6:         V ← V ∪ {I_i}
7:         E ← E ∪ {(q,i)}
8:         w_qi ← 1 / |M_i|
```

### 6.4 公式

$$V=\Bigl\{I_i\in K' : \bigl|I_i^{LG}\bigr|>\tau_m=500\Bigr\},\qquad
w_{ij}=\frac{1}{\#\text{inliers}(i,j)}$$

- **为什么用基础矩阵而不是单应？** 乱序采集包含大幅位姿变化（前进、侧移、抬升），不是同一平面的近似；单应会误杀。
- **为什么边权取倒数？** 内点越多 → 几何关系越强 → 权重越**小**。这样后面"最短路径"就等于"最可靠的位姿链"（⑥ 的 Dijkstra 直接受益），而 ④ 的累计连通度只要把权重再倒数回来即可。
- 代码里存的是 $1/n\_matches$，选窗时 `1.0 / edge_weight` 还原为内点数——**与论文式 (1) 完全等价**，只是存法不同。

### 6.5 超参

| 参数 | 论文 | 代码 | 备注 |
|---|---|---|---|
| $\tau_m$ | 500 | `min_matches_edge = 500` | 一道很硬的门槛：宁可少建边 |
| 单对耗时 | ≈4 ms | — | 最贵，所以只跑 5 对 |

### 6.6 贡献归属

- ★ **本文贡献**：把"匹配可靠度"编码成**图上的边权**，让 ④ 选窗和 ⑥ 回环判据都能直接复用同一个数。这是"一个量服务两个决策"的设计。
- ☆ 借用：LightGlue（Lindenberger 2023）、RANSAC + 基础矩阵（经典）。

---

## 7 ④ 选窗：在协视图上贪心扩张（论文式 1）

### 7.1 干什么

决定这次 BA **优化哪些帧**。不是"最近 20 帧"，而是"和当前集合连接最强的那个"，一次加一个，加到 20 个。

### 7.2 输入 / 输出

- 输入：③ 更新后的加权协视图
- 输出：变量集合 $S$（只决定"算谁"，不动任何数值）

### 7.3 算法

```
1: S ← {I_q}
2: while |S| < N:
3:     for each i ∈ N(S) \ S:            # 只访问 S 的邻居
4:         c_i ← Σ_{j∈S} 1 / w_ij        # 还原成内点数再累加
5:     i* ← argmax c_i
6:     S ← S ∪ {i*}
```

### 7.4 公式

$$c(I_i,S_t)=\sum_{I_j\in S_t}\#\text{inliers}(i,j)
\qquad
\textbf{(1)}\quad S_{t+1}=S_t\cup\Bigl\{\operatorname*{argmax}_{I_i\in\mathcal{M}\setminus S_t}c(I_i,S_t)\Bigr\}$$

- 这是论文**唯一编号的正文公式**，也是全篇最该记住的一条。
- 它与 SLAM 的"时间窗"是**互斥**的两种假设：时间窗假设"相邻帧相关"，式 (1) 假设"共视才相关"。乱序输入下，前者直接失效。
- 复杂度：每步只遍历 $\mathcal{N}(S)\setminus S$，而论文实测**每个节点平均只有约 10 条边**，所以一次 argmax 几千次加法，可忽略。

### 7.5 超参

| 参数 | 论文 | 代码 | 备注 |
|---|---|---|---|
| $N$ | 20 | `num_prev_keyframes_miniba_incr = 5`（增量部分） | 全窗 20，增量 5 |
| 节点平均度 | ≈10 | — | 贪心便宜的根本原因 |

### 7.6 贡献归属

- ★ **本文贡献**：式 (1)。这是 C1（"efficient retrieval of well-connected subsets"）的核心。
- ☆ 借用：协视图本身来自 ORB-SLAM（Mur-Artal 2015）。
- **消融最关键**：`NoKFSelection` 26.29 → **24.56 dB（−1.73 dB）**，全场掉得最多。也就是说"选对窗"比"匹配得更准"更影响最终画质——因为窗选错，后面怎么优化都救不回。

---

## 8 ⑤ 局部 BA：唯一真正解位姿的一步

### 8.1 干什么

在选出的 20 帧 + 约 1 万个路标上解一次带鲁棒核的最小二乘，用 Schur 补把问题压到"只解相机"。

### 8.2 输入 / 输出

- 输入：$S$ 内 20 个关键帧；按观测数采样的 $K=10000$ 个路标；观测 $u_{ij}$
- 输出：$T_i$（位姿）与 $X_j$（三维点）——**重建本体**

### 8.3 算法

```
 1: S ← SelectKeyframes(I_q, G, N=20)              # ④
 2: L ← 多项式采样(K=10000, 权重 = 每个路标的观测数)
 3: for it in 1..iters:
 4:     r_ij ← u_ij − π(T_i, X_j)                  # 重投影残差
 5:     w_ij ← Huber(‖r_ij‖)                       # 鲁棒权重
 6:     J    ← ∂π/∂ξ  (CUDA kernel，解析雅可比)     # 6 维扰动，非 9 维
 7:     H ← Σ JᵀWJ ;  g ← Σ JᵀWr                  # 分块：B(相机) / C(点) / E(交叉)
 8:     H_ii ← H_ii · (1 + λ)                      # LM 阻尼
 9:     S ← B − E C⁻¹ Eᵀ                           # 稠密 Schur 补
10:     δ_cam ← Cholesky(S) 解
11:     δ_pt  ← C⁻¹(−g_pt − Eᵀ δ_cam)              # 回代，C 的 3×3 逆逐点并行
12:     接受/拒绝 ← torch.where(新残差 < 旧残差)     # 无数据相关分支
```

### 8.4 公式

**目标函数**（带 Huber 鲁棒核与阻尼项）：

$$\min_{\{T_i\},\{X_j\}}\ \sum_{(i,j)\in\mathcal{O}}\rho\!\left(\bigl\|u_{ij}-\pi(T_i,X_j)\bigr\|^2\right)+\lambda\|\delta\|^2$$

**正规方程与 Schur 补**：

$$\begin{bmatrix}B&E\\ E^{\top}&C\end{bmatrix}\begin{bmatrix}\delta_{\text{cam}}\\ \delta_{\text{pt}}\end{bmatrix}
=-\begin{bmatrix}g_{\text{cam}}\\ g_{\text{pt}}\end{bmatrix}
\;\Longrightarrow\;
S=B-EC^{-1}E^{\top},\quad
\begin{cases}
S\,\delta_{\text{cam}}=-g_{\text{cam}}+EC^{-1}g_{\text{pt}}\\
\delta_{\text{pt}}=C^{-1}\bigl(-g_{\text{pt}}-E^{\top}\delta_{\text{cam}}\bigr)
\end{cases}$$

- $C$ 是**块对角**（每个路标 3×3），所以 $C^{-1}$ 可以逐点并行求逆，不需要解大矩阵；
- 剩下要解的只有 $S$，维度 = 相机数 × 6 = 120 —— 这就是"把问题压到只解相机"的含义；
- 论文明确说：在这个规模下**稠密 Schur 补比稀疏求解器更快**（稀疏结构带来的访存代价超过收益）。

**位姿雅可比**（左扰动，$\xi=(v,\omega)\in\mathbb{R}^6$）：

$$\frac{\partial p_{\text{cam}}}{\partial \xi}=\bigl[\,I\ \big|\ -[p_{\text{cam}}]_\times\,\bigr]\in\mathbb{R}^{3\times6}$$

注意这里是 **6 维扰动**（不是把旋转写成 3×3 再用 9 维），所以雅可比是 3×6，量级小一半。

**se(3) 指数映射**（把 6 维增量变回变换矩阵）：

$$R=I+A[\omega]_\times+B[\omega]_\times^2,\qquad V=I+B[\omega]_\times+C[\omega]_\times^2,\qquad t=Vv$$

$$A=\frac{\sin\theta}{\theta},\qquad B=\frac{1-\cos\theta}{\theta^2},\qquad C=\frac{\theta-\sin\theta}{\theta^3},\qquad \theta=\|\omega\|$$

$\theta<10^{-5}$ 时三个系数退化为泰勒展开，避免除零——这是 CUDA 核里的显式分支。

### 8.5 超参

| 参数 | 论文 | 代码 | 备注 |
|---|---|---|---|
| $N$ | 20 | 20 | DROID-SLAM 只用 5 帧时间窗 |
| 路标数 $K$ | 10000 | 10000 | 按观测数多项式采样，**固定规模**才能上 CUDA Graph |
| 固定帧 $n_f$ | 4 | `fixed_masks` | 参与损失、不参与优化 |
| 迭代次数 | 10（表 15） | `self.iters` | 每关键帧固定 |
| 阻尼 | LM | `H.diagonal().mul_(1 + lm)` | 单调增，接受则减半（标准 LM） |

### 8.6 实现细节（这一节是"工程贡献"的落点）

- **自写 CUDA 核**：残差与雅可比在 CuPy `RawModule` 里 JIT 编译（无 setup.py，运行时编译），避免 PyTorch 逐算子调度。
- **Schur 消元不落 CPU**：`S = B − E C⁻¹ Eᵀ` 全程在显存里完成，用 `cholesky_ex` + 两次回代；$C^{-1}$ 走自定义 `inv_3x3` 逐点并行。
- **整轮 LM 压成一张 CUDA Graph**：`torch.cuda.make_graphed_callables(..., 2)` 捕获，省掉每轮几十次 kernel 启动开销。代价是**循环里不许有数据相关的分支**，所以接受判据用 `torch.where` 而不是 `if`（见 `mini_ba.py:391`）。
- **`atomicAdd` 抢槽位**：新观测写入时用 `add_obs_to_existing_kernel` 原子抢 slot，避免"先数一遍再分配"的主机同步。
- **每个路标只被固定子集观测**：这是论文自己强调的关键——否则观测数随 $N$ 二次增长，$N{=}20$ 时稠密 Schur 补在显存上不可能。

### 8.7 贡献归属

- ★ **本文贡献**：把"固定路标子集 + 自写 GPU 核 + 稠密 Schur 补 + CUDA Graph"组合起来，让 $N{=}20$ 第一次在交互延迟内可解。这是 C1 后半句 "allowing GPU-friendly local reconstruction" 的实际内容。
- ○ 教科书：LM、Huber、Schur 补、Cholesky、se(3) 指数映射（Barfoot / 视觉 SLAM 十四讲级别）。
- ☆ 借用：Lambda Twist P3P（自举用）、DA3（深度初始化）、30 次迭代的高斯优化沿用 Meuleman 2025。
- **加速比（表 15）**：g2o 461.4 ms → CudaBA 78.1 → On-The-Fly 79.7 → **本文 15.9 ms**，快 5–29 倍。

---

## 9 ⑥ 回环修正：簇级判定 + 免优化的 Sim(3) 传播

### 9.1 干什么

判定"两簇其实重叠"，然后**不重跑全局 BA**：用一次焊接窗口 BA 求出 Sim(3)，沿主干按测地距离插值摊开。

### 9.2 输入 / 输出

- 输入：新边让两簇之间突然出现跨簇匹配；协视图 $G$
- 输出：全图位姿与高斯的同一份 Sim(3) 修正

### 9.3 算法

```
 1: K'_2 ← 取 2K' 个最佳匹配                        # 2×5=10，均匀时两簇各 5
 2: 以"连通意义上最远"的两帧为种子 → 簇 C1, C2
 3: if 加权图距(C1,C2) > min_loop_size(=10) and 小簇有效匹配对 > τ_v(=2):
 4:     焊接 BA：从锚点向两侧各扩 N/2 帧，固定老簇 n_f 帧
 5:     SE(3)：锚点"初始位姿 → 重优化位姿"的刚体差
 6:     s_Δ ← 路标深度变化；S_Δ ← Sim(3)(s_Δ, R_Δ, t_Δ)
 7:     backbone ← Dijkstra(G, 不含回环边)          # 回环边此时还没入图
 8:     for k in backbone:
 9:         α_k ← d_geo(锚点,k) 归一化                # 0 = 老簇固定端，1 = 被优化锚点
10:         T_k ← ΔT^{α_k} ∘ T_k                     # 旋转 6D 插值、尺度对数插值
11:     高斯：g ← s_Δ R_Δ g + t_Δ（跟随出生关键帧）
12:     把回环匹配写回 G，供后续局部 BA 使用
```

### 9.4 公式

**Sim(3) 沿主干的比例插值**：

$$\Delta T_\alpha=\exp\bigl(\alpha\cdot\log\Delta T\bigr),\qquad \alpha\propto d_{\text{geo}}$$

- $\alpha$ 沿主干从 0（老簇固定端）单调增到 1（被优化的锚点），按**测地距离**比例取；
- 旋转用 **6D 表示**插值（Zhou 2019）再正交化，避免四元数双覆盖的符号跳变；
- 尺度用**对数插值**：$s_\alpha=s_\Delta^{\alpha}$，保证"先放大再缩小"可逆、不出现负尺度。

**Sim(3) 作用在高斯上**（代码 `ba_problem.py:1043-1049`）：

$$p'=s_\Delta R_\Delta\,p+t_\Delta,\qquad \sigma'=s_\Delta\,\sigma,\qquad R'=R_\Delta R$$

即位置做相似变换、尺度乘 $s_\Delta$、旋转左乘 $R_\Delta$。**每个高斯记着"谁生了我"**，所以它跟着出生关键帧走，不需要重新优化。

为什么非 Sim(3) 不可：

- 单目（含深度先验）的**尺度不可观测**，回环两侧的尺度天然不一致；
- 只用 SE(3)，回环在数学上就**不闭合**——接缝处的点会错开一个比例，表现为"两片场景错层"。

### 9.5 超参

| 参数 | 论文 | 代码 | 备注 |
|---|---|---|---|
| 回环判据 | 跳数 $>\tau_{lc}=5$ | 加权图距 $>10$（`min_loop_size`） | **论文与代码不一致，见 §12** |
| 小簇最少对 | $\tau_v=2$ | — | 防止单对误检 |
| 焊接窗口 | 两侧各 $N/2$ 帧 | — | 保证 N 帧 BA 的均衡 |
| 固定帧 | 老簇 $n_f$ 帧 | — | 整体约束路标移动 |
| 回环最佳匹配 | $2K'$ | `topk(n_matches, n*2)` | 均匀时两簇各 $K'$ |

### 9.6 贡献归属

- ★ **本文贡献**（C2 全部）：
  1. **簇级判据**替代时间戳——乱序下唯一可行的判法；
  2. **焊接 BA** 搬到簇语境（原概念属 ORB-SLAM 的焊接窗口）；
  3. **免优化的位姿+高斯更新**——这是"优化-free"三个字的实际含义：不算全局 BA，只插值。
- ☆ 借用：焊接窗口（Mur-Artal 2015）、SE(3)/Sim(3)、6D 旋转（Zhou 2019）。
- ○ 教科书：Dijkstra。
- 消融：`NoGraphProp` 25.81 dB（这项主要比速度：PyPose PGO 要跑优化，本文只插值）；`NoGSUpdate` 25.45 dB（回环后不搬高斯 −0.84 dB）。

---

## 10 ⑦ 高斯落位：分块深度对齐 + 渐进式层级

### 10.1 干什么

位姿定下来才投放高斯：分块估深度尺度与偏移，再把屏幕上看不见的小高斯合并掉、把用不上的搬到 CPU。

### 10.2 输入 / 输出

- 输入：位姿 $T_i$、三角化稀疏点、单目深度（Depth Anything 3）
- 输出：新高斯（绑定到 $I_q$）；更新后的层级 $H$

### 10.3 算法

```
 1: D ← DA3(I_q)                              # 逆深度
 2: d_tri ← 三角化得到的稀疏深度
 3: for tile k in 4×4 网格:                    # 分块
 4:     最小二乘解  D ≈ s_k·d_tri + o_k        # 每块一组 (s,o)
 5: 双线性上采样 (s,o) 到原分辨率 → 逐像素对齐
 6: 按对齐后的深度生成高斯，绑定到 I_q
 7: S_i ← f·r_i/z_i                            # 屏幕尺寸
 8: if 计数(S_i < τ_l=0.5 px) > max(10%, 200000):
 9:     kNN 取最近 4 个 → 去重 → 合并成父节点
10: 高斯优化 30 次迭代（采样邻近视角）
11: 不可见叶子卸载到 CPU RAM
```

### 10.4 公式

**分块深度对齐**：

$$d_{\text{mono}}(u)\approx s_k\,d_{\text{tri}}(u)+o_k,\qquad u\in\text{tile}_k$$

- $s_k,o_k$ 在**每块内**最小二乘求出，再线性插值上采样到逐像素（$s,o$ 随空间缓变，插值安全）；
- 论文的对照是"整图只算一组 $s,o$"（Kerbl 2024 的做法）——一张图里有近景桌面和远景墙面，单一线性关系必然两头不讨好；
- 代码里对齐是在**逆深度**空间做的（`mono_idepth`），数值更稳。

**屏幕尺寸与合并判据**：

$$S_i=\frac{f\,r_i}{z_i},\qquad \text{merge if } S_i<\tau_l=0.5\ \text{px}$$

- $f$ 焦距、$r_i$ 高斯最大尺度、$z_i$ 深度。$S_i<0.5$ px 意味着这个高斯在屏幕上**连一个像素都占不满**，它的贡献可以近似为零。
- 合并触发（论文）：亚像素高斯**超过总量 20%** 才做，避免每帧都合并。
- 合并触发（代码）：`max(10%, 200000)` 双门限。**20 万这个绝对量是给小场景的"关闭开关"**——小场景永远达不到 20 万，所以层级根本不建。这是论文没写、但很关键的一处工程决定。

### 10.5 超参

| 参数 | 论文 | 代码 | 备注 |
|---|---|---|---|
| 分块网格 | 未给 | `depth_grid_size = 4`（4×4） | 4×4 是效果/开销折中 |
| $\tau_l$ | 0.5 px | `hierarchy_screen_size_threshold` | 亚像素门限 |
| 合并触发 | >20% | `max(ratio=0.1, min_count=200000)` | **不一致，见 §12** |
| 合并方式 | KL 散度、4 个一组 | kNN(4) + `opacity·screen_size²` 加权平均 | **不一致，见 §12** |
| 重载门限 | $\tau_h=1$ | — | 贡献超过 1 就换回细高斯 |
| 每帧迭代 | 30 | `30` | 实时采集时改为异步跑 |

### 10.6 贡献归属

- ★ **本文贡献**（C3 全部）：按**屏幕尺寸**建层级、父子 ID 挂载、双门限触发、**异步搬移**（避免影响延迟）、双缓冲（避免漫游卡顿）。
- ★（未列入三条贡献的补充）：**分块 scale-offset 深度对齐**——论文 Sec. 5.2 自述为本文做法，但没进贡献列表。答辩如果被问"还有没有别的贡献"，这一条可以答。
- ☆ 借用：高斯生成方式（Meuleman 2025）、KL 合并（Kerbl 2024）、kNN（Papantonakis 2024）、DA3（Lin 2025）。
- 消融：`NoLoDepAlgn` 25.41 dB（−0.88 dB）；表 16：中等场景峰值高斯数**降 38%**，大场景 CityWalk **不开层级直接 OOM**（24 GB 卡在 20% 进度前就爆）。

---

## 11 复杂度与时间账

### 11.1 每步的量级

| 步骤 | 复杂度 | 实测/估算 | 是否随帧数增长 |
|---|---|---|---|
| ① 检索 | $O(\lvert\mathcal{M}\rvert\cdot4096)$ | ≈1 ms/1000 对 | 线性（可忽略） |
| ② 粗筛 | $O(K\cdot n_{kp})$ | ≈0.5 ms×20 = 10 ms | 常数 |
| ③ 精验 | $O(K'\cdot n_{kp})$ | ≈4 ms×5 = 20 ms | 常数 |
| ④ 选窗 | $O(N\cdot N\cdot \bar d)$，$\bar d\approx10$ | 可忽略 | 常数 |
| ⑤ 局部 BA | $O(\text{iter}\cdot(N^3 + N K))$ | **40.5 ms** | 常数（$N,K$ 都固定） |
| ⑥ 回环 | 焊接 BA + Dijkstra + 逐帧插值 | 摊销 **121.6 ms**（仅回环时） | 近线性 |
| ⑦ 落位+层级 | $O(\lvert\mathcal{G}\rvert)$ 屏幕尺寸 + 触发式 kNN | 余量 | 线性 |

### 11.2 单帧预算对得上 324 ms 吗

论文实测：**平均 324 ms、最大 386 ms**（MipNeRF360，定义为 "the time from start of processing to readiness for next frame"）。

$$27.8\ (\text{前端}) + 40.5\ (\text{BA}) + 121.6\ (\text{回环摊销}) + \underbrace{\sim134}_{\text{高斯落位/优化/层级}} \approx 324\ \text{ms}$$

三项显式数字加起来只有 190 ms，差额在"新增高斯 30 次迭代优化 + 层级更新"——PPT P4 上写的就是这个账。**这也是最容易被追问的地方**：看到 27.8 和 40.5 会以为"那不就 68 ms 吗"，答案在剩下的 134 ms。

#### 11.2.1 本次复现的逐帧实测

robotlab 采集的 85 张 1280×1707 手机竖拍图（ds=1）。在训练循环每帧末尾（`train.py` 的 `update_pbar`）
打一行时间戳，纯净跑一遍（不落中间快照），全程 **54.5 s**：

| 口径 | 数值 | 说明 |
|---|---|---|
| 单帧延迟中位 | **87 ms** | 多数帧只做匹配 + 少量优化 |
| p90 | 756 ms | 触发了 BA 或回环修正的帧 |
| 最慢一帧 | 1.72 s | 全程没有一帧超过 2 s |
| 84 帧主体 | 26.1 s | 平均 **311 ms/帧**，与论文 324 ms 同一量级 |
| 收尾重试搁置帧 | 28.4 s | 与"边拍边看"无关，可后置 |
| 全程 | 54.5 s | 即 `metadata.json` 的 `time` |

**口径陷阱**：用 `metadata.json` 的 `time / num keyframes` = 54.103 s / 81 = **668 ms** 会低估实时性，
因为那个数把收尾重试的 28 秒也摊进了每一帧——它回答的是"整段录完平均每帧摊多少"，
而不是论文定义的 `time from start of processing to readiness for next frame`。
单帧延迟应看**中位与分位数**。峰值 1.72 s 高于论文 386 ms，原因是像素量约为论文典型场景的 4 倍。

### 11.3 三个"实时"必须分清

| 说法 | 数字 | 归谁 |
|---|---|---|
| **重建实时**（immediate） | 324 ms/帧（峰值 386） | **本文的卖点** |
| 渲染实时 | 480 fps（2.1 ms/帧） | 3DGS 表示自带，**不是本文贡献** |
| 全流程 | 85 张 1 min 17 s vs 离线 8 min 50 s（6.9×） | 本文（含所有优化） |

---

## 12 论文公式 vs 开源代码：三处不一致

答辩高风险区。如果在台上被问"你跑的和论文一样吗"，照下面答——说明我们是**照代码核对过**的。

| # | 项 | 论文 | 代码 | 怎么答 |
|---|---|---|---|---|
| C.1 | VPR 模型 | MixVPR（Ali-bey 2023） | `MegaLoc` | 许可原因替换，同为 4096 维余弦检索，用法一致 |
| C.2 | 回环判据 | 跳数 $>\tau_{lc}=5$ | 加权图距 $>10$（`min_loop_size`） | 同为"两簇在图上够不够远"，度量不同 |
| C.3 | 层级合并 | >20% 触发；KL 散度、4 个一组 | `max(10%, 200000)` 双门限；kNN(4) + `opacity·screen_size²` 加权平均 | 双门限给小场景留了关闭开关；20 万门槛意味着小场景根本不建层级 |

补充第 4 处（表述级）：论文 Sec. 4.1 称"两级匹配"，但实际是 **VPR → XFeat → LightGlue** 三级。`TwoLevelMatching` 这个函数名把后两级算作一级。

---

## 13 一页速查

| 步 | 一句话 | 核心公式 | 关键超参 | 贡献 |
|---|---|---|---|---|
| ① | 全局描述子比余弦，取 20 | $s=d_q\cdot d_i$ | $K{=}20$ | ★ 把 VPR 接进在线第一级 |
| ② | XFeat 比点数，20→5 | $K'=\text{top-}5\lvert I^{BF}_i\rvert$ | $K'{=}5$ | ★ 三级速度/质量级联 |
| ③ | LightGlue+RANSAC，>500 建边 | $\lvert I^{LG}\rvert>\tau_m,\ w=1/\lvert M\rvert$ | $\tau_m{=}500$ | ★ 内点倒数当边权 |
| ④ | 按累计内点数贪心扩到 20 | 式 (1) $S_{t+1}=S_t\cup\arg\max c$ | $N{=}20$ | ★ 式 (1)，替代时间窗 |
| ⑤ | LM+Schur，只解相机 | $\min\sum\rho+r$；$S=B-EC^{-1}E^{\top}$ | $N{=}20,K{=}10^4,n_f{=}4$ | ★ 固定观测子集 + CUDA Graph（15.9 ms） |
| ⑥ | 图距>10 触发，Sim(3) 摊开 | $\Delta T_\alpha=\exp(\alpha\log\Delta T)$ | 图距 10、$\tau_v{=}2$ | ★ 簇级判定 + 免优化传播 |
| ⑦ | 分块对深度，按屏幕尺寸合并 | $d\approx s_k d_{tri}+o_k$；$S_i{=}fr_i/z_i$ | 4×4、$\tau_l{=}0.5$ | ★ 渐进式层级（峰值高斯 −38%） |

**三句可以背下来的话：**

1. 七步里真正解位姿的只有第 ⑤ 步，其余六步都在为它准备条件或修正它的结果。
2. 本文的三条贡献分别落在 **①②③④（匹配+选窗）**、**⑥（回环）**、**⑦（层级）**；⑤ 的贡献不是算法而是"让它算得动"。
3. LM、Schur 补、Huber、Dijkstra 全是教科书内容——**本文没有提出新数学，提出的是新的问题分解方式**。

---

## 资料来源

| 来源 | 位置 |
|---|---|
| 论文原文 PDF | `论文/i3dgs_SIGGRAPH2026_原文.pdf` |
| 英文原文文本 | `论文/_src_clean.txt`（arXiv HTML 全文提取，本文公式与引文据此校对） |
| 中文全译 | `论文/i3dgs_论文中文全译.md` |
| 英文原句对照 | `论文/i3dgs_关键步骤英文原文对照.md` |
| 开源代码 | `172.88.88.12:/home/hjadmin/zw/ecnu/ecnv-cg/i3dgs` |
| 典型片段 | `scene/vpr_model.py`、`scene/scene_model_loop_closure.py:179-240`、`poses/ba_problem.py:496-509 / 647-672 / 1040-1062`、`poses/mini_ba.py:284-300 / 391 / 506`、`scene/mini_ba_utils.cu:133 / 186`、`scene/mono_depth.py:258-282`、`scene/hierarchy_cut.py:90` |

> 公式均为本人根据论文正文（Sec. 4–7）、附录 Algorithm 1 与开源代码整理，其中 ⑤ 的 Schur 分块、位姿雅可比、se(3) 指数映射三式的具体写法以 `mini_ba_utils.cu` 的实现为准（论文只给结论式）。数值一律标注出处：表 6 / 表 15 / 表 16 为论文原表，其余为本次复现实测。
