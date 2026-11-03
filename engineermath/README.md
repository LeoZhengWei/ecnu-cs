# 特征脸算法笔记

配合《高级工程数学》第 7 题的推导文档。主文件是 `eigenfaces_algorithm.tex`，正文在 `chapters/`，导言区宏包与自定义命令在 `preamble/`。编译器为 XeLaTeX（`ctexart`）。

## 环境

本机没有系统级 TeX Live，且当前账号不能免密 `sudo`，因此装在用户目录中的 TinyTeX（TeX Live 2026）。可执行文件链接到 `~/.local/bin`，该目录已在默认 `PATH` 中。

| 项目 | 位置 |
| --- | --- |
| 发行版 | `~/.TinyTeX` |
| `xelatex`、`latexmk`、`tlmgr` | `~/.local/bin` |
| 宏包镜像 | 清华大学 CTAN 镜像 |

已安装并在本文档中用到的宏包：`ctex`、`xeCJK`、`fontspec`、`zhnumber`、Fandol 字体、`geometry`、`amsmath`、`amsfonts`（提供 `amssymb`）、`booktabs`、`xcolor`、`hyperref`、`latexmk`。

`hyperref` 的链接色写成了 `blue!60!black`，这是 `xcolor` 的混色语法，所以 `preamble/packages.tex` 在 `hyperref` 之前加载了 `xcolor`。

### 从头安装

需要 `perl`、`curl`（或 `wget`）和 `xz`。

```bash
curl -fsSL "https://yihui.org/tinytex/install-bin-unix.sh" | sh
export PATH="$HOME/.local/bin:$PATH"

tlmgr option repository https://mirrors.tuna.tsinghua.edu.cn/CTAN/systems/texlive/tlnet
tlmgr update --self
tlmgr install ctex fandol booktabs geometry amsfonts xecjk zhnumber cjk \
  latexmk fontspec xcolor etoolbox hyperref amsmath environ trimspaces ulem
```

安装脚本会把命令链接到 `~/.local/bin`。新开的终端若找不到 `xelatex`，把下面这一行写入 `~/.bashrc`：

```bash
export PATH="$HOME/.local/bin:$PATH"
```

以后缺宏包时：

```bash
tlmgr install 宏包名
```

## 编译

在本目录执行：

```bash
latexmk eigenfaces_algorithm.tex
```

`.latexmkrc` 已指定 XeLaTeX，并会按目录和交叉引用自动多编几遍。成功后得到 `eigenfaces_algorithm.pdf`。

不用 `latexmk` 时需要手动编两次，否则目录是空的：

```bash
xelatex eigenfaces_algorithm.tex
xelatex eigenfaces_algorithm.tex
```

清理中间文件和 PDF：

```bash
latexmk -c
```

这些产物已写入 `.gitignore`，不进入版本库。
