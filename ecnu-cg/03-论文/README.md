# 论文原文、中文全译与论文精要

> ⚠️ **本目录的 PDF 是 Andreas Meuleman 等人的原创论文，著作权归原作者与 ACM 所有，不是本仓库作者的成果。**
>
> 本仓库作者只做了三件事：① 阅读并复现该论文（成果见 `../01-报告/`，只跑通 garden 一个场景）；
> ② 把原文逐段翻译成中文（`i3dgs_论文中文全译.md`，非官方译本）；
> ③ 按"读稿/讲稿/答辩"三种用途压缩出一份精要（`i3dgs_论文精要.md`）。
> 收录原文仅为课程学习与对照研读，**不主张任何署名权**。

| 文件 | 说明 |
|---|---|
| `Meuleman2026_i3dgs_SIGGRAPH2026_原文.pdf` | 原作者版 arXiv PDF（v1），14.8 MB，含全部彩图 |
| `i3dgs_论文中文全译.md` | 本仓库作者翻译的中文全译，约 2.77 万字，摘要→结论 + 附录 A/B |
| `i3dgs_论文精要.md` | 约 2300 字（全译的 1/12），22 张表，见下节 |

## 三份文档怎么用

| 你要做的事 | 打开哪个 |
|---|---|
| 查某个数字、某句原文怎么写的 | `i3dgs_论文中文全译.md`（逐段对照原文，公式用可读记法） |
| 10 分钟过一遍这篇论文讲什么 | `i3dgs_论文精要.md`（一句话→三约束→四机制→关键数字→消融→局限） |
| 上台前练逐页口播 | `../04-讲稿/演讲讲稿与答辩手册.md` |
| 完全不懂图形学时先看懂原理 | `../04-讲稿/零基础白话讲义.md`（30 条术语白话对照） |

`i3dgs_论文精要.md` 的取舍标准是"少了它这篇论文就讲不完整，或被问到会答不上来"，
因此**删去了相关工作、致谢、逐条公式推导与参考文献**，保留四节机制、两组消融、三张对比表的数字，
并额外记录了 **论文正文与官方代码的 3 处不一致**（MegaLoc 替换 MixVPR、层级合并双门限 0.1 + 20 万、
"两级匹配"的表述差异）——这 3 条在原文与任何二手资料里都找不到，是读代码核出来的。

## 论文信息（原始出处）

**Immediate 3D Gaussian Splat Reconstruction of Unordered Input with Global Consistency**
（面向无序输入的即时三维高斯泼溅重建，具备全局一致性）

| 项 | 内容 |
|---|---|
| 作者 | **Andreas Meuleman**\*、**Linus Franke**\*（共同一作）、Boris Zhestiankin、Camille Montemagni、**George Drettakis** |
| 单位 | Inria, Université Côte d'Azur（法国）；EPFL（瑞士）；Inria, Université de Rennes（法国） |
| 发表 | SIGGRAPH Conference Papers '26，2026 年 7 月 19–23 日，美国洛杉矶（ACM TOG） |
| DOI | [10.1145/3799902.3811167](https://doi.org/10.1145/3799902.3811167) |
| arXiv | [2607.14481v1](https://arxiv.org/abs/2607.14481)（cs.CV，2026-07-16 提交） |
| ISBN | 979-8-4007-2554-8/2026/07 |
| 官方代码 | <https://github.com/graphdeco-inria/i3dgs>（Inria GraphDeco，作者团队开源） |

## 著作权与使用范围

- **论文原文版权归 ACM 及上述各位作者所有**。本仓库收录这份 PDF，仅用于课程作业中的学习、
  研读与复现说明，**不作任何再分发、商用或其他用途**；任何引用请引用 ACM 出版版本，
  不要引用本仓库。
- `i3dgs_论文中文全译.md` 是本人在课程研读过程中**自行翻译的非官方译本**，仅供自己与同组同学
  对照原文阅读，不代表作者观点，**一切技术细节以原文为准**。
- 若原作者或出版方对上述文件的收录有异议，请在 Issues 中指出，会立即移除。
- 作者姓名、单位、DOI、ISBN 均按原文首页逐字核对；如仍有笔误，欢迎指正。

## 引用格式

```bibtex
@inproceedings{meuleman2026i3dgs,
  title     = {Immediate {3D} Gaussian Splat Reconstruction of Unordered Input with Global Consistency},
  author    = {Meuleman, Andreas and Franke, Linus and Zhestiankin, Boris and
               Montemagni, Camille and Drettakis, George},
  booktitle = {ACM SIGGRAPH 2026 Conference Papers},
  series    = {SIGGRAPH '26},
  year      = {2026},
  doi       = {10.1145/3799902.3811167},
  publisher = {ACM},
  address   = {New York, NY, USA}
}
```

## 译者说明

1. 译文依据 arXiv HTML 版（v1）**逐段翻译**，章节编号、图号、表号均沿用原文，可与 PDF 对照阅读。
2. 公式以可读的纯文本记法写出（如 `w_ij = 1/#inliers(i,j)`），未使用 LaTeX 排版。
3. 关键超参数已还原为具体数值（N=20、K=10000、τ_lc=5、τ_m=500 等），原文 HTML 中被折叠的公式
   亦已恢复。
4. 参考文献列表未逐条翻译，需要时请对照原文。
5. 【译注】标记处为译者补充，非原文内容。
