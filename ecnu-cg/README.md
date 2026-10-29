# 即时三维高斯泼溅重建（i3dgs）复现

> 课程：图形和视觉前沿技术（ECNU）　|　类型：SIGGRAPH 顶会论文复现 + 演示
>
> 论文：**Immediate 3D Gaussian Splat Reconstruction of Unordered Input with Global Consistency**
> 作者：**Andreas Meuleman**、**Linus Franke**（共同一作）、Boris Zhestiankin、Camille Montemagni、**George Drettakis**（Inria GraphDeco / Université Côte d'Azur 等）
> 会议：SIGGRAPH 2026（ACM TOG）
> 官方代码：<https://github.com/graphdeco-inria/i3dgs>
> 论文原文：<https://arxiv.org/abs/2607.14481>

本仓库只收录**复现产出的关键内容**（报告、PPT、讲稿、演示脚本、实验数据、演示视频）。
可运行的代码工作区（含 CUDA 扩展构建产物、数据集、模型权重，约 8 GB）不入库，
放置在 `172.88.88.14:/home/hjadmin/zhengwei/i3dgs-demo`，详见 `05-DEMO/`。

> **署名声明**：本仓库是课程复现作业。上面的论文及其 PDF（`03-论文/`）是 **Andreas Meuleman 等作者的
> 原创成果**，著作权归原作者与 ACM；本仓库作者只负责阅读、复现与中文翻译，**不主张论文的任何署名权**。

---

## 一、仓库结构

```
ecnu-cg/
├── README.md                本文件
├── 交付说明.md               交付清单、现场演示建议、提交前检查项
├── 01-报告/                  课程报告（docx + HTML/Markdown 源 + 渲染图 + 生成脚本）
├── 02-PPT/                   15 页汇报 PPT + slidep 源文件 + 预览图 + 讲稿要点
├── 03-论文/                  论文原文 PDF + 中文全译 + 论文精要 + 版权说明
├── 04-讲稿/                  逐页口播讲稿 + 答辩手册 + 零基础白话讲义
├── 05-DEMO/                  一键脚本、使用说明、本地补丁包、漫游演示视频（garden + 自拍场景）
└── 06-实验数据/              运行摘要、metadata、完整训练日志
```

| 目录 | 关键文件 | 说明 |
|---|---|---|
| `01-报告/` | `图形和视觉前沿技术报告-i3dgs复现.docx` | 六章，软件学报排版（单栏 / 宋体 10.5pt / 行距 1.25） |
| | `report.html`、`tools/embed_img.py` | 报告排版源；docx 中图片以 base64 内联，故有此脚本 |
| `02-PPT/` | `PPT-即时三维高斯泼溅重建.pptx` | 15 页（要求 ≤15），含摘要页与贡献点页；第 7 页为**算法地图**（七步重建链：算法的公式与在位姿重建中的作用），第 15 页为**附录**（七个步骤的真实源码片段 + 文件行号） |
| | `slides/01..15.slide`、`DESIGN.md`、`STORY.md` | slidep DSL 源、视觉规范、叙事结构 |
| `03-论文/` | `Meuleman2026_i3dgs_SIGGRAPH2026_原文.pdf` | **原作者版** arXiv v1 PDF（14.8 MB），著作权归 Andreas Meuleman 等作者与 ACM，仅供课程学习，本仓库不主张署名权 |
| | `i3dgs_论文中文全译.md` | 约 2.77 万字，摘要→结论 + 附录 A/B，表格数字全部保留 |
| | `i3dgs_论文精要.md` | 约 2300 字（全译的 1/12），22 张表：一句话→三约束→四机制→关键数字→消融→局限→必答题 |
| | `README.md` | 作者、DOI、arXiv 编号、BibTeX、使用范围声明，以及"三份文档怎么用"的对照表 |
| `04-讲稿/` | `演讲讲稿与答辩手册.md` | 15 页逐页口播词（主讲合计 13:55，附页不计时）+ 20 个答辩问题 |
| | `零基础白话讲义.md` | 30 条术语白话对照，无公式版 5 分钟讲法 |
| `05-DEMO/` | `run_all_14.sh` | 新机器一键流程：自动挑空闲 GPU → 重建 → 渲染 |
| | `patch-i3dgs.zip` | 3 处 `[LOCAL PATCH]`（权重离线化、dtype、VPR 替换） |
| | `garden_ds2_walkthrough.mp4` | 真实重建结果 149 帧漫游视频（现场断网兜底） |
| | `self-capture-robotlab/` | **自拍场景实录**：85 张乱序手机照片、无位姿无内参 → 55 s 重建 → 240 帧新轨迹漫游视频 + 证据图 + 局限说明 |
| `06-实验数据/` | `SUMMARY_garden.txt`、`metadata_ds2*.json`、`train_eval_ds2.log` | 原始数据，报告与 PPT 中的数字均出自此处 |

---

## 二、复现结果

garden 场景，**185 帧无序输入、不提供 COLMAP 位姿先验**，2 倍降采样（648×420），单张 RTX 4090：

| 指标 | 数值 |
|---|---|
| PSNR / SSIM / LPIPS | **25.95 dB / 0.786 / 0.206** |
| 位姿误差（对 COLMAP 真值） | **R° 0.155 / t 1.888** |
| 关键帧数 / 高斯数 | 156 / 362 044 |
| 流式重建墙钟耗时 | **62.7 s**（185 帧，2.95 帧/秒） |
| 峰值显存 | 19.5 GB |
| 渲染速度 | 约 480 帧/秒 |

> `SUMMARY_garden.txt` 中的 `train_sec=104` 是脚本整段计时，含启动时的模块初始化与 JIT 编译；
> 62.7 s 是纯流式重建（位姿求解 + 增量建图）的时间。

论文在 A100、完整分辨率、有序输入下报 26.29 dB / 1:17；本机为半分辨率、每 8 帧留 1 帧作测试集，
差距来自分辨率与测试协议，量级一致。

### 自拍场景（真实使用验证，非数据集）

用自己手机拍的 **85 张乱序照片**（微信下载顺序，**未排序**）、**不给位姿、不给内参**，直接跑通：

| 指标 | ds=2 | **ds=1（成片）** |
|---|---|---|
| 关键帧 / 输入 | 76 / 85 | **81 / 85** |
| 高斯数 | 471 902 | **1 857 881** |
| 流式重建耗时 | 30.9 s | **54.9 s** |
| 峰值显存 | 4.86 GB | 8.45 GB |
| 焦距初值 → 收敛 | 896（= 0.7×宽）→ **1209.8** | 同 |

漫游视频、证据图与完整说明见 [`05-DEMO/self-capture-robotlab/`](05-DEMO/self-capture-robotlab/)。

> 两点边界：这个场景**没有 COLMAP 真值**，只能评图像质量，**不能给位姿误差**；PSNR 也不能与上表或论文横向比较。
> 它的价值是一次**真实的乱序输入**验证——garden 用的是有序视频序列，只能说明"未使用 COLMAP 位姿先验"，
> 不构成"能处理乱序"的证据。这次照片的到达顺序就是微信 `MsgID` 的随机顺序，属于真·无序。

---

## 三、环境

| 项 | 值 |
|---|---|
| 机器 | `172.88.88.14`，Ubuntu 22.04 |
| GPU | 4 × RTX 4090 24 GB（训练用单卡） |
| 驱动 / CUDA | 550.163.01 / CUDA 12.6（nvcc 12.6） |
| Python | 3.10，Miniconda 装于 `/datafast`（根分区仅剩 16 GB） |
| PyTorch | 2.7.1 + cu126 |
| 编译目标 | `TORCH_CUDA_ARCH_LIST=8.9`（换到 40 系必须从 12.0 改回 8.9） |

目录分工：代码在 `~/zhengwei/i3dgs-demo`，环境 / 数据 / 权重 / 结果在 `/datafast/zhengwei-i3dgs`。

## 四、怎么跑

```bash
ssh hjadmin@172.88.88.14
cd /home/hjadmin/zhengwei/i3dgs-demo
bash scripts/run_all_14.sh --ds 4      # 自动挑最空闲的 4090；4 倍降采样几分钟内出结果
```

完整参数、12 个已知坑、交互式漫游（`gaussianviewer.py server` + 本地客户端）见
`05-DEMO/使用说明（172.88.88.14）.md`。

---

## 五、边界与诚实声明

- 只跑通 **garden 一个场景**，且是 **2 倍降采样**；未复现论文全部 4 个数据集的完整指标表。
- 精度差论文最主要的两个来源：分辨率减半、每 8 帧抽 1 帧作测试集（论文为全分辨率、标准划分）。
- 位姿精度在**宽基线场景**（MipNeRF360，APE 8.5）有优势；在 TUM 这类小基线基准上
  **不如专用 SLAM（DROID-Splat 1.0）**，论文表 4 自身即可佐证。
- 代码实现与论文正文有一处不一致：视觉位置识别实际用 **MegaLoc**（`scene/vpr_model.py` 注释写明
  因许可原因替换 MixVPR）；层级合并触发条件为 `ratio=0.1` 且 `min_count=200000` 双门限。
- 论文原文 PDF 已随仓库提供（`03-论文/`），**版权归 ACM 及各位作者所有**，仅用于课程学习与研读，
  不作再分发或商业用途；中文全译为自行翻译的非官方译本，技术细节以原文为准，详见
  `03-论文/README.md`。
- 报告与 PPT 的署名为 **郑炜、周子懿**，指导教师 **DeepSeek**。

## 六、引用

```bibtex
@article{meuleman2026i3dgs,
  title   = {Immediate {3D} Gaussian Splat Reconstruction of Unordered Input with Global Consistency},
  author  = {Meuleman, Andreas and others},
  journal = {ACM Transactions on Graphics (SIGGRAPH)},
  year    = {2026}
}
```

本项目基于官方 i3dgs 代码（Inria GraphDeco，非商业研究许可）与 3DGS 原始实现（Inria / MPII）。
