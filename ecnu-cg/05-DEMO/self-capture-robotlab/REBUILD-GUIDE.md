# 从 85 张照片到可旋转 3D 场景 —— 手动操作手册

> 目标机器：`172.88.88.14`（4× RTX 4090），用户 `hjadmin`，密码 `hjjs@2025`
> 这份手册里的命令**按顺序复制粘贴即可**，每段都能单独跑。

---

## 一、现在的状态（2026-10-30 16:58 实测，不是猜的）

| 项 | 位置 | 状态 |
|---|---|---|
| 照片 85 张 / 28 MB | `/datafast/zhengwei-i3dgs/data/robotlab/images/` | **已在服务器**，与本地 `sha256` **85/85 全一致** |
| 训练成品 | `/datafast/zhengwei-i3dgs/results/robotlab_ds1/` | 在：`hierarchy.ply` 490 MB + `metadata.json` + `video/rendered_path.mp4` |
| 重建脚本 | `/home/hjadmin/zhengwei/i3dgs-demo/scripts/run_scene14.sh` | 在 |
| 轨迹渲染脚本 | `/home/hjadmin/zhengwei/i3dgs-demo/scripts/orbit_render.py` | 在 |
| 深度先验权重 | `/datafast/zhengwei-i3dgs/weights/DA3-BASE` | 在 |
| 磁盘 | `/datafast` 1.8 T，已用 58%，剩 704 G | 够 |

**结论：照片不用再传了。** 下面第二节是"万一要重传"的命令，平时跳过。

---

## 二、登录与重传（可选）

### 2.1 登录

```bash
ssh hjadmin@172.88.88.14
# 密码 hjjs@2025
```

### 2.2 重传照片（只有在你要覆盖/换一批图时才需要）

在**本地 Windows** 的 Git Bash 里跑：

```bash
cd "C:/Users/13205/WorkBuddy/2026-10-16-19-11-43"
export SSH_PW='hjjs@2025'

# 生成 85 行 local=remote 清单
"D:/python/python3/python.exe" - <<'PY'
import os
src = r"user_demo/upload/images"
dst = "/datafast/zhengwei-i3dgs/data/robotlab/images"
lines = [f'{os.path.abspath(os.path.join(src, f))}={dst}/{f}'
         for f in sorted(os.listdir(src)) if f.lower().endswith(".jpg")]
open("_up_manifest.txt", "w", encoding="utf-8").write("\n".join(lines))
print(len(lines), "files")
PY

"C:/Users/13205/.workbuddy/binaries/python/envs/default/Scripts/python.exe" \
  "C:/Users/13205/.workbuddy/skills/ssh-fleet-ops/scripts/ssh_put.py" \
  172.88.88.14 --user hjadmin --manifest _up_manifest.txt --keep-crlf
```

`ssh_put.py` 会自己算双端 `sha256` 并打印校验结果，看到 `ALL OK` 即成功。
**`--keep-crlf` 不能省**：脚本默认把 CRLF 转成 LF，会把 JPG 改坏。

---

## 三、训练：照片 → 3D 高斯

### 3.0 环境变量（每开一个新终端都要先跑这段）

```bash
PROJ=/home/hjadmin/zhengwei/i3dgs-demo
BIG=/datafast/zhengwei-i3dgs
ENV=$BIG/envs/i3dgs
PY=$ENV/bin/python
WEIGHTS=$BIG/weights
RESULTS=$BIG/results

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.9"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
```

### 3.1 一条命令跑完（推荐，脚本自带尺寸检查 + 自动挑最空的卡）

```bash
bash $PROJ/scripts/run_scene14.sh \
  --scene-path $BIG/data/robotlab \
  --name robotlab \
  --ds 1
```

- `--ds 1` = 原分辨率 1280×1707，1.86 M 高斯，8.5 GB 显存，93 秒跑完
- `--ds 2` = 半分辨率，430K 高斯，约 31 秒，画质明显糊一档
- 不加 `--ds` 默认 2；脚本发现最空卡 < 14 GB 时会自动跳到 `ds=3`

⚠ **这条命令会 `rm -rf $RESULTS/robotlab_ds1`**，把现有那 490 MB 成品删掉重来。
如果只是想重渲视频，不要跑它，直接跳第四节。

### 3.2 分步手动跑（想看完整日志时用）

```bash
cd $PROJ/i3dgs

$PY -u train.py \
  -s $BIG/data/robotlab \
  -m $RESULTS/robotlab_ds1 \
  --downsampling 1 \
  --da3_model_dir $WEIGHTS/DA3-BASE
```

想同时出 PSNR / SSIM / LPIPS（留出 1/6 帧做测试）：

```bash
$PY -u train.py \
  -s $BIG/data/robotlab \
  -m $RESULTS/robotlab_ds1 \
  --downsampling 1 \
  --da3_model_dir $WEIGHTS/DA3-BASE \
  --test_hold 6 --test_frequency 15 --eval_poses
```

### 3.3 训练完能拿到什么

```
$RESULTS/robotlab_ds1/
├── hierarchy.ply     490 MB   ← 真正的 3D 模型（带层级 + kf_id 属性）
├── metadata.json      76 KB   ← 相机位姿、内参 f、每个关键帧的统计
├── sparse/0/                  ← 训练相机（cameras.bin / images.bin）
└── video/rendered_path.mp4    ← ⚠ 这是"照片重放"，不是漫游，见第五节
```

---

## 四、渲染漫游视频（这一步最容易做错）

**不要**用 `render_path.py --render_path $RESULTS/robotlab_ds1/sparse/0`。
那是**训练时那批相机**，渲染相机和拍照相机完全重合，出来的视频就是把 81 张原照片按顺序播一遍——第一版就栽在这里。

正确做法是从 `metadata.json` 造一条**训练时从未到过**的新轨迹再渲染：

```bash
$PY $PROJ/scripts/orbit_render.py \
  --model_path $RESULTS/robotlab_ds1 \
  --out_dir   $RESULTS/robotlab_ds1/orbit \
  --mode smooth --n_frames 240 --fps 30
```

- `--mode smooth`（默认）沿原有相机方位角走一圈、半径与高度做 9 帧滑动平均，最稳
- `--mode orbit` 走弧线 + 推近 + 抬升（吊臂镜头）
- `--mode dolly` 沿视线前后推拉
- `--mode parallax` 同一位置左右各偏 5cm，出一对视差图，用来证明"这不是照片拼接"

先看效果可以只渲几张静态帧，几十秒出结果：

```bash
$PY $PROJ/scripts/orbit_render.py \
  --model_path $RESULTS/robotlab_ds1 --out_dir /tmp/chk \
  --mode smooth --preview 6
```

> **安全边界**：新相机必须夹在训练相机包络内（实测中位偏移 7.6 cm 最好）。
> 抬到 44° 俯视时主体几何还对，但天花板和背景墙会糊成一片——只有单侧观测，高斯被拉长。
> 这是论文自己承认的覆盖不足局限，不是 bug。

---

## 五、把 3D 模型拿回本地，变成能转的网页

服务器没有图形界面，必须在本地看。

### 5.1 取回（490 MB，sftp 约 5–15 分钟）

```bash
cd "C:/Users/13205/WorkBuddy/2026-10-16-19-11-43"
export SSH_PW='hjjs@2025'

"C:/Users/13205/.workbuddy/binaries/python/envs/default/Scripts/python.exe" \
  "C:/Users/13205/.workbuddy/skills/ssh-fleet-ops/scripts/ssh_get.py" \
  172.88.88.14 --user hjadmin \
  -m "/datafast/zhengwei-i3dgs/results/robotlab_ds1/hierarchy.ply=3dgs/robotlab_hierarchy.ply"
```

只想快点看，也可以先取 `metadata.json` + 视频，模型放到最后再拉。

### 5.2 清洗成标准 3DGS ply

`hierarchy.ply` 带 `kf_id`、`child_0..3` 这些层级字段（共 66 个属性），
三方工具（Blender 插件、splat 转换器）按 62 个标准属性读，直接喂会错位。

```bash
cd "C:/Users/13205/WorkBuddy/2026-10-16-19-11-43/3dgs"

"D:/python/python3/python.exe" clean_ply.py \
  robotlab_hierarchy.ply robotlab_3dgs.ply \
  --crop-pct 98 --scale-ratio 30
```

- `--crop-pct 98`：按到场景中心距离剪掉最远 2%，i3dgs 有约 5% 离群点，不剪会糊满屏
- `--scale-ratio 30`：剪掉尺度大于中位数 30 倍的 floater

### 5.3 转 splat 并生成可旋转网页

```bash
"D:/python/python3/python.exe" ply2splat.py robotlab_3dgs.ply robotlab.splat
"D:/python/python3/python.exe" build_viewer.py robotlab.splat robotlab_3dgs.html --title "机器人场景 · 3DGS"
```

双击 `robotlab_3dgs.html` 就能在浏览器里拖拽旋转。
1.86 M 高斯全量网页会很大（garden 10 万高斯已经是 1.2 MB），建议限一下：

```bash
"D:/python/python3/python.exe" ply2splat.py robotlab_3dgs.ply robotlab_lite.splat --max 300000
"D:/python/python3/python.exe" build_viewer.py robotlab_lite.splat robotlab_lite.html
```

### 5.4 另一条路：Blender

`robotlab_3dgs.ply` 可以直接用 Blender + **KIRI 3DGS Render** 插件打开，能自由转视角、
也能打关键帧出动画。细节见同目录 `3DGS模型怎么打开.md`。

---

## 六、五个必须知道的坑

1. **尺寸必须一致**：`image_dataset.py` 只读第一张图的宽高，全流程单一相机模型。
   混进竖拍/异尺寸的图会错位甚至报错。那 85 张都是 1280×1707，别再往里加别的图。
2. **`--ds 1` 出图更清晰，但 PSNR 反而更低**（19.47 vs ds2 的 20.76）。
   这是评测协议问题（低分辨率下高频细节少、更容易拟合），**不要用 PSNR 论证分辨率收益**，用肉眼/SSIM。
3. **`run_scene14.sh` 最后会调 `render_path.py --render_path sparse/0`**，产出的
   `video/rendered_path.mp4` 是照片重放。想看真漫游必须另外跑 `orbit_render.py`。
4. **HEIC 不支持**。白名单只有 `.jpg/.png/.jpeg/.webp`，iPhone 原图要先转。
5. **显存**：全分辨率 eval 阶段约 8.5 GB。如果别的用户在卡上占了 9 GB+，会 OOM，
   脚本会自动挑最空的卡，但 eval 阶段仍可能撞上——换个时段或换 `--ds 2`。

---

## 七、全流程速查（复制即用）

```bash
# ① 登录
ssh hjadmin@172.88.88.14

# ② 环境
PROJ=/home/hjadmin/zhengwei/i3dgs-demo; BIG=/datafast/zhengwei-i3dgs
ENV=$BIG/envs/i3dgs; PY=$ENV/bin/python; WEIGHTS=$BIG/weights; RESULTS=$BIG/results
export CUDA_HOME=/usr/local/cuda PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.9" HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1

# ③ 训练（会删除旧结果！）
bash $PROJ/scripts/run_scene14.sh --scene-path $BIG/data/robotlab --name robotlab --ds 1

# ④ 漫游视频
$PY $PROJ/scripts/orbit_render.py --model_path $RESULTS/robotlab_ds1 \
    --out_dir $RESULTS/robotlab_ds1/orbit --mode smooth --n_frames 240 --fps 30

# ⑤ 回本地后：清洗 → splat → 网页
cd 3dgs
"D:/python/python3/python.exe" clean_ply.py robotlab_hierarchy.ply robotlab_3dgs.ply
"D:/python/python3/python.exe" ply2splat.py robotlab_3dgs.ply robotlab_lite.splat --max 300000
"D:/python/python3/python.exe" build_viewer.py robotlab_lite.splat robotlab_lite.html
```
