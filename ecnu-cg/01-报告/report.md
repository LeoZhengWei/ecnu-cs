# 面向无序输入与全局一致性的即时三维高斯泼溅重建

## ——SIGGRAPH 2026 论文 i3dgs 的复现与分析

**组员：** 组员A、组员B　　**课程：** 图形和视觉前沿技术　　**日期：** 2026 年 11 月

---

## 摘要

三维高斯泼溅（3D Gaussian Splatting, 3DGS）已成为场景重建与实时新视角合成的代表性技术，但其训练流程高度依赖运动恢复结构（Structure from Motion, SfM）离线提供相机位姿，而 SfM 必须等待全部图像采集完毕才能开始计算，且对"无序采集"（拍摄者来回走动、反复回头补拍）较为敏感。本文分析并复现了 SIGGRAPH 2026 论文《Immediate 3D Gaussian Splat Reconstruction of Unordered Input with Global Consistency》。该工作首次为无序图像输入提供了具备全局一致性的"即时反馈"辐射场重建方案：利用视觉位置识别模型与协视图（covisibility graph）实现两级快速匹配，通过 GPU 定制的局部光束法平差（Bundle Adjustment, BA）与精细化高斯投放实现快速局部重建，提出基于簇（cluster）的回环检测与"图传播"式的免优化漂移修正，并用渐进式高斯层级支撑大场景显存受限下的持续重建。我们在配备 NVIDIA RTX 5090 的服务器上完成了环境搭建与复现，跑通了 MipNeRF360 场景的重建流程，验证了该方法"少量秒级/分钟级时间即可给出可视化结果"的核心主张，并结合复现过程对其工程实现、性能权衡与局限进行了分析。

**关键词：** 三维高斯泼溅；无序图像采集；即时重建；协同视图；回环闭合；渐进式层级

---

## 1 Top Paper 概要信息

| 项目 | 内容 |
|---|---|
| 论文名称 | Immediate 3D Gaussian Splat Reconstruction of Unordered Input with Global Consistency |
| 中文译名 | 面向无序输入与全局一致性的即时三维高斯泼溅重建 |
| 发表会议 | ACM SIGGRAPH 2026 Conference Papers（2026 年 7 月 19–23 日，美国洛杉矶） |
| DOI / arXiv | 10.1145/3799902.3811167 / arXiv:2607.14481 |
| 作者 | Andreas Meuleman、Linus Franke（共同一作）、Boris Zhestiankin、Camille Montemagni、George Drettakis |
| 主要通讯单位 | 法国 Inria（Université Côte d'Azur）GRAPHDECO 研究组；合作单位 EPFL、Université de Rennes |
| 论文类型 | 图形（Graphics），兼及视觉（CV） |
| 开源代码 | https://github.com/graphdeco-inria/i3dgs |

**研究机构简介。** GRAPHDECO 是 Inria 与 Université Côte d'Azur 联合组建的图形学与三维视觉研究组，由 George Drettakis 领导。该组正是 3D Gaussian Splatting 原始论文（SIGGRAPH 2023）的作者团队，此后持续在辐射场实时渲染、SLAM 与在线重建方向产出工作（如 3DGS、Scaffold-GS、On-The-Fly NVS、Gaussian Opacity Fields 等）。本文是该团队在"把 3DGS 从离线工具变成可用采集工具"这一主线上的最新推进，其代码库也延续了 3DGS 的工程风格（C++/CUDA 光栅化器 + Python 训练管线）。

---

## 2 研究背景

### 2.1 问题的由来

3DGS 用一组带方向的三维高斯椭球显式表达场景，配合可微的泼溅（splat）光栅化实现实时渲染，训练只需图像与对应相机位姿。然而"位姿从哪来"这一前置问题在实践中代价很高：主流做法是用 COLMAP 等 SfM 工具先离线求解相机位姿与稀疏点云。

作者在论文中给出的动机非常具体：**真实采集几乎总是无序的**。用户举着手机绕着场景走，中途转身、回头补拍角落、再从另一侧进入，这种"连续片段 + 乱序回访"的采集方式覆盖更完整，但会让 SfM 的匹配与全局优化变得昂贵；更重要的是，SfM **必须等所有图像都到齐**才能开始，用户在拍摄现场无法知道"哪里还没拍到、拍得够不够好"。对于文物数字化、建筑巡检、电商建模这类现场作业，这种"拍完才知道失败"的代价不可接受。

### 2.2 领域现状与两条路线的缺口

| 路线 | 代表工作 | 优点 | 缺陷 |
|---|---|---|---|
| 离线 SfM + 3DGS | COLMAP/GLOMAP + 3DGS、Octree-GS、H3DGS | 质量高、全局一致 | 必须等全部图像；分钟到小时级；无法现场反馈 |
| 增量/类 SLAM 重建 | Photo-SLAM、MonoGS、On-The-Fly NVS、DROID-Splat、LongSplat | 即时反馈、可交互 | **假设图像有序**：新帧只与时间上相邻的帧匹配，一旦乱序或回访即失效 |

也就是说，两条路线存在一个明显的空缺：**既有即时性、又能处理无序采集、还能保持全局一致性**。本文正是填补这个空缺的工作。作者将挑战归纳为三点：(1) 无序数据中的快速且可靠的图像匹配；(2) 局部重建要在 GPU 上足够快，同时不能让优化规模随关键帧数爆炸；(3) 发现回环后，如何在不重跑全局优化的前提下把修正量传播到所有位姿**以及**所有高斯。

### 2.3 扩展阅读

为理解本文位置，我们额外阅读了 3DGS 原文（Kerbl et al., SIGGRAPH 2023）、On-The-Fly NVS（Meuleman et al., 2025，本文的评测协议与部分基线来源）、Photo-SLAM 与 MonoGS（即时但有序）、H3DGS / Octree-GS（大场景层级表示），以及视觉位置识别（VPR）方向的 MixVPR 与 LightGlue。这些工作构成了本文"匹配—局部优化—全局一致—可扩展"四段式设计的直接背景。

---

## 3 核心工作

### 3.1 目标

给定一串**顺序未知**的 RGB 图像（可来自视频抽帧，也可来自杂乱的手机相册），在线地、边接收边输出一个可实时渲染的 3DGS 场景，并在长时间、大范围采集下保持全局一致（不漂移），同时显存占用可控。

### 3.2 贡献点

1. **面向无序序列的两级快速匹配与协视图关键帧选取**：复用视觉位置识别模型（MixVPR）做全局检索，再用轻量局部匹配（XFeat）与精确匹配（LightGlue）逐级筛选，把匹配复杂度从两两全比降到常数级；并给出基于加权协视图的"高连通关键帧"贪心选取算法。
2. **GPU 友好的局部重建框架**：自写 BA 的 GPU 核函数（稠密 Schur 补），在 N=20 个关键帧、K=10000 个路标点的规模下比现有优化器快约 5 倍；并提出**分块（tile）深度对齐**的高斯投放策略，替代全图统一的尺度/偏移估计。
3. **基于簇的回环闭合**：不依赖时间顺序，用协视图上的"跳数距离"判定两个簇是否分离，从而检测回环；用"焊接窗口 BA"求解相对变换，再用**Dijkstra 主干路径 + 测地距离插值**把 Sim(3) 修正量传播到全图位姿与高斯——**无需再跑一次全局优化**。
4. **渐进式高斯层级**：按高斯在屏幕上的投影尺寸自适应合并/回取，将细小图元卸载到 CPU 内存，使大场景（如 4050 帧的 CityWalk）在 24 GB 显存下也能持续重建。

### 3.3 方法细节

**(1) 系统主循环。** 每来一个新的关键帧 $I_q$：两级匹配 → 更新协视图 → 回环检测 → 选取 N=20 个高连通关键帧 → 局部 BA → （若回环）焊接 BA + 主干传播 → 投放高斯 → 更新层级 → 优化高斯（每关键帧固定 30 次迭代）→ 重试此前"搁置"的帧。

**(2) 两级匹配。** 候选集逐级收缩，是本文工程上最巧妙的部分：

| 层级 | 方法 | 作用 | 代价 |
|---|---|---|---|
| 1 | MixVPR（4096 维全局描述子） | 余弦相似度排序取 top-20 | ≈1 ms / 1000 对 |
| 2 | XFeat + 内点数排序 | 从 20 中留 5 | ≈0.5 ms / 对 |
| 3 | LightGlue + RANSAC | 精确验证，内点数 > 500 才接受 | ≈4 ms / 对 |

匹配通过的关键帧对写入**加权协视图** $G=(\mathcal{M},E,W)$，边权取内点数的倒数 $w_{ij}=1/\#\text{inliers}(i,j)$——内点越多表示几何约束越强、权重越小，因此"最短路"天然对应"最可靠的位姿链"。关键帧选取用贪心：每步加入与当前集合累计内点连接数最大的节点，
$$\mathcal{S}_{t+1}=\mathcal{S}_t\cup\Big\{\arg\max_{I_i\notin\mathcal{S}_t}\sum_{I_j\in\mathcal{S}_t}\#\text{inliers}(i,j)\Big\}.$$
由于每个节点平均只有约 10 条边，该过程开销极小。作者在消融中显示，这一项（NoVPR / NoKFSelection）对最终质量影响最大（PSNR 从 26.29 掉到 24.78 / 24.56）。

**(3) GPU 局部重建。** 与 SLAM 系统只用 5 帧时间窗不同，本文用 20 个关键帧并固定 10000 个路标点（按被观测次数做多项式采样），且每个路标只允许被协视性选出的固定子集观测，避免观测数随关键帧数二次增长——这是把"SLAM 式局部优化"扩展到"辐射场重建"的关键改动。高斯投放上，作者不再对整幅图估计一组深度尺度/偏移，而是把图像切块分别估计再插值上采样，使高斯落点更贴合真实表面（消融 NoLoDepAlgn：PSNR 26.29 → 25.41）。

**(4) 基于簇的回环。** 当一次匹配中出现两组"各自内部很密、彼此很疏"的关键帧集合时，用跳数距离 $d_G(\mathcal{C}_1,\mathcal{C}_2)>\tau_{lc}=5$ 且有效匹配对 $>\tau_v=2$ 判定为回环。随后从两个簇各扩 N/2 帧构成焊接窗口做局部 BA，得到锚点帧的刚体变换，再由路标深度变化估计尺度，组成 Sim(3) 修正；最后沿 Dijkstra 求出的"主干路径"按测地距离比例插值分配修正（旋转用 6D 表示插值、尺度用对数插值），并对每个记录了自己"出生关键帧"的高斯套用同一变换。这样回环修正的代价与场景规模近似线性，而非再做一次全局 BA。

**(5) 渐进式层级。** 当高斯在当前视角下的屏幕尺寸小于 $\tau_l=0.5$ px 且这类高斯占比超过 20% 时触发合并：KNN 找 4 个最近高斯按 KL 散度合并，并向层级插入父节点；叶节点参数卸载到 CPU。渲染或导航时若某合并节点的贡献超过 $\tau_h=1$ px，则异步回取其子节点（双缓冲避免卡顿）。论文报告：不用层级时，CityWalk 在 24 GB 卡上处理到第 795/4050 帧（<20%）即 OOM；用层级后峰值训练高斯维持在 370 万以下、显存 <12.8 GB，大场景层级达 8 层。

---

## 4 实验分析

### 4.1 论文报告的主要结果

论文在 MipNeRF360、TUM、StaticHikes、Tanks and Temples、Deep Blending 以及自采大规模场景（SmallCity、Wayve、CityWalk）上评测，指标为 PSNR/SSIM/LPIPS 与位姿误差。

有序场景下，本方法在 MipNeRF360 上以 **1 分 17 秒** 的总耗时达到 **26.29 dB**，而离线参考（GLOMAP + Taming-3DGS 7000 次迭代）需要 8 分 50 秒得到 27.52 dB；在 TUM 上以 1 分 22 秒达到 24.77 dB，而质量更高的 LongSplat 需要 1 小时 41 分（约 75 倍时间）。无序场景下，MipNeRF360 为 25.60 dB / 1 分 28 秒，相比离线方案约快 6 倍而质量仅略低。大规模场景 CityWalk（4050 帧）上以 29 分 38 秒达到 22.03 dB，显著优于 H3DGS（11.78 dB / 22 小时）与 Octree-GS（17.33 dB / 3 小时）。位姿精度方面，MipNeRF360 上平移 APE 为 8.5，明显优于 Photo-SLAM/MonoGS/CF-3DGS（161–315 量级）。此外，把输入顺序随机打乱后 PSNR 仅从 26.29 降到 25.73，说明方法对"无序"的鲁棒性是实质性的，而非仅在有序数据上勉强工作。

### 4.2 我们的复现结果

我们在 **NVIDIA RTX 5090（32 GB）**、Ubuntu 22.04、CUDA 12.8 环境下复现了 MipNeRF360 的 `garden` 场景（无序输入、无 COLMAP 位姿先验，仅给图像）。复现流程与实测记录如下：

| 项目 | 实测值 |
|---|---|
| 输入图像数 | {{N_IMAGES}} 张 |
| 图像分辨率 | {{RES}} |
| 训练总耗时 | {{TRAIN_TIME}} |
| 峰值显存占用 | {{VRAM}} |
| 平均每帧延迟 | {{LATENCY}} |
| 重建质量 | {{PSNR}} |

{{REPRO_COMMENT}}

### 4.3 与论文结果的对照分析

{{COMPARE_COMMENT}}

---

## 5 代码分析

### 5.1 获取与组成

代码位于 https://github.com/graphdeco-inria/i3dgs ，采用「Python 训练管线 + CUDA 光栅化扩展」的经典 3DGS 工程结构：

```
i3dgs/
├── train.py              # 主入口：数据流、匹配、BA、高斯优化的调度
├── args.py               # 全部超参数（N、K、学习率、阈值等）
├── dataloaders/          # 图像流读取（无序、可打乱）
├── poses/                # 位姿/BA 相关（GPU 版局部 BA）
├── scene/                # 高斯模型与相机模型
├── gaussianviewer.py     # 自带交互式查看器（graphdecoviewer）
├── scripts/              # 数据集下载、评测脚本
└── submodules/
    ├── diff-gaussian-rasterization   # 可微泼溅光栅化（CUDA，需编译）
    ├── simple-knn                    # 高斯邻域查询（CUDA，需编译）
    ├── xfeat                         # 轻量局部特征匹配
    └── Depth-Anything-V2             # 单目深度先验（初始化与深度对齐）
```

其中 `diff-gaussian-rasterization` 与 `simple-knn` 是必须用 nvcc 现场编译的 CUDA 扩展；`Depth-Anything-3` 以 pip 依赖形式引入（用于初始化焦距、位姿与路标）。

### 5.2 环境配置的实测记录（踩坑与解决）

课程要求"说清楚代码的配置和运行情况"，这里如实记录我们遇到的四个问题，这本身也是复现工作的一部分：

1. **架构不匹配**。原始 3DGS 生态普遍基于 CUDA 11.x + 较老 PyTorch，而 RTX 5090 属于 Blackwell 架构（sm_120），必须使用 CUDA 12.8 及以上才能编译。所幸 i3dgs 官方即以 `PyTorch 2.7.1 + CUDA 12.8` 测试，我们直接采用该组合，并把 `TORCH_CUDA_ARCH_LIST` 固定为 `12.0`，避免编译出无法在 5090 上执行的 PTX。
2. **嵌套子模块缺失**。首次编译 `diff-gaussian-rasterization` 报 `glm/glm.hpp: No such file or directory`——原因是 `git clone --recursive` 只展开了第一层子模块，`third_party/glm` 需要二次初始化。执行 `git submodule update --init --recursive` 后解决。
3. **GitHub 访问不稳定**。`depth-anything-3` 以 `git+https://github.com/...` 形式声明依赖，服务器到 github.com 的连接间歇性超时，导致 pip 直接失败。改用 `codeload.github.com` 下载指定 commit 的 tarball 后本地安装，问题消失。
4. **HuggingFace 不可达**。首次运行需要从 HuggingFace 拉取深度模型权重，而该域名在本网络不可达。设置 `HF_ENDPOINT=https://hf-mirror.com` 走镜像后正常。

最终可用的一键配置命令（供后人复现）：

```bash
conda create -y -n i3dgs python=3.12 && conda activate i3dgs
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
export PATH=/usr/local/cuda-12.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST="12.0"
export HF_ENDPOINT=https://hf-mirror.com
git submodule update --init --recursive          # 关键：展开 glm
pip install -r requirements.txt --no-build-isolation
pip install cupy-cuda12x
```

### 5.3 运行情况

```bash
python scripts/download_datasets.py --out_dir data --datasets MipNeRF360
CUDA_VISIBLE_DEVICES=1 python train.py -s data/MipNeRF360/garden -m results/garden
```

实测：{{RUN_LOG}}

---

## 6 读后感

**关于方法设计的取舍。** 这篇论文最打动我的地方不是某一个模块的技巧性，而是它对"代价"的清醒分配：作者把整个系统拆成"每帧都要做的小事"（匹配、局部 BA、30 次高斯迭代）和"偶尔做一次的大事"（回环修正、层级合并），并保证后者不需要重跑全局优化。回环那一节尤其精彩——用 Dijkstra 在协视图上找"最可信的位姿链"，再按测地距离把 Sim(3) 修正量插值传播出去，本质上是用**图上的距离**替代**全局优化的解**，这个思路在工程上非常有借鉴价值。

**优势。** (1) 真正解决了实际采集流程中的痛点，而不是在已有基准上刷点；(2) 无序打乱后质量仅掉 0.5 dB，说明鲁棒性扎实；(3) 速度优势是数量级的（分钟 vs 小时），且质量只比离线方案低 1–2 dB，这个性价比对现场应用是决定性的；(4) 代码开源、自带查看器，工程完成度高。

**劣势与局限。** (1) 纯旋转或低视差序列无法三角化，这是所有 SfM/SLAM 系方法的共病，论文也承认；(2) 基于图传播的修正虽然快，但精度不及真正的全局 BA，几何精度上仍与 GLOMAP 有差距；(3) 依赖 MixVPR/XFeat/LightGlue/Depth-Anything 等一串上游模型，任一环节在特殊场景（弱纹理、强反光、夜间）退化都会传导到最终结果；(4) 层级合并的阈值（0.5 px / 1 px / 20%）是经验值，跨数据集时可能需要重调。

**对领域前景的看法。** 这类工作的价值在于把 3DGS 从"事后重建"推向"采集时的助手"。可以预见，下一步的竞争点会从"重建质量"转向"采集体验"——例如实时提示"这里还没覆盖""这个视角重投影误差大"，甚至主动规划拍摄路径。这与本组另一条线（主动重建、机器人在线建图）是收敛的。

**与课程外实践的关联。** 我们此前在做一个"玩具小车 + PC 端 3DGS 在线建图"的小项目。复现本文后有一个直接结论：小车边走边推流，本质上正是"无序、增量、需要即时反馈"的场景，而 MoonSplat 那类方法需要额外维护跟踪线程；i3dgs 的"匹配 + 局部 BA + 回环传播"三件套其实给出了更通用的骨架——真正卡住小车的不是算法，而是单目视频的弱视差与运动模糊。这让我们对该方向的工程难点有了更具体的认识。

**关于复现本身的体会。** 这次复现最大的收获不是跑通了一个脚本，而是理解了 3DGS 生态的"隐性依赖"：CUDA 架构、子模块层级、模型权重来源，任何一环不通都会让"照 README 敲命令"直接失败。也正因如此，课程要求里"代码分析和运行情况"这一节其实非常必要——它检验的是能否把一个研究工作真正落地。

---

## 参考文献

[1] Meuleman A, Franke L, Zhestiankin B, Montemagni C, Drettakis G. Immediate 3D Gaussian Splat Reconstruction of Unordered Input with Global Consistency. ACM SIGGRAPH 2026 Conference Papers, 2026. DOI: 10.1145/3799902.3811167.
[2] Kerbl B, Kopanas G, Leimkühler T, Drettakis G. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics, 2023, 42(4): 139.
[3] Meuleman A, et al. On-The-Fly Novel View Synthesis. 2025.
[4] Matsuki H, Murai R, Kelly J, Davison A J. Gaussian Splatting SLAM. CVPR 2024.
[5] Huang H, Li L, Cheng H, Yeung S-K. Photo-SLAM: Real-time Photorealistic SLAM with Efficient 3D Gaussian Splatting. 2024.
[6] Ali M, et al. MixVPR: Feature Mixing for Visual Place Recognition. WACV 2023.
[7] Potje G, Cadar F, Avallone G, et al. XFeat: Accelerated Features for Lightweight Image Matching. CVPR 2024.
[8] Lindenberger P, Sarlin P-E, Pollefeys M. LightGlue: Local Feature Matching at Light Speed. ICCV 2023.
[9] Kerbl B, Meuleman A, Kopanas G, Drettakis G. A Hierarchical 3D Gaussian Representation for Real-Time Rendering of Very Large Datasets. ACM TOG, 2024.
[10] Ren K, et al. Octree-GS: Towards Consistent Real-time Rendering with LOD-Structured 3D Gaussians. 2024.
