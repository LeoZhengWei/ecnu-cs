#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 i3dgs 训练过程中 --save_every 落下的中间快照逐个渲染成图。

为什么用"该时刻最后一个关键帧的相机"渲染：
    这正是拍摄者按下快门那一刻看到的画面 —— 也就是论文里 immediate 的定义。
    换成固定视角反而拍不出早期快照（那时那块区域还没建）。

用法：
    python snapshot_render.py \
        --prog_dir /datafast/zhengwei-i3dgs/results/robotlab_prog/progress \
        --log /datafast/zhengwei-i3dgs/logs/prog.log \
        --out_dir /datafast/zhengwei-i3dgs/results/robotlab_prog/render
"""
import argparse
import json
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import torch

PROJ = "/home/hjadmin/zhengwei/i3dgs-demo/i3dgs"
sys.path.insert(0, PROJ)

from scene.scene_model import SceneModel   # noqa: E402
from utils import focal2fov                # noqa: E402


def ply_n_vertices(path):
    """逐行读 ply 头，拿顶点数。hierarchy.ply 属性多，header 常超过 8 KB。"""
    n = -1
    with open(path, "rb") as fh:
        for _ in range(200):
            line = fh.readline()
            if not line:
                break
            if line.startswith(b"element vertex"):
                n = int(line.split()[2])
            if line.strip() == b"end_header":
                break
    return n


def parse_frame_times(log_path, total):
    """从 tqdm 进度条里抠出每一帧的累计墙钟秒数。"""
    times = {}
    if not log_path or not os.path.isfile(log_path):
        return times
    raw = open(log_path, "rb").read().decode("utf-8", "ignore")
    pat = re.compile(r"(\d+)/" + str(total) + r" \[(\d+):(\d+)(?::(\d+))?<")
    for seg in re.split(r"[\r\n]+", raw):
        m = pat.search(seg)
        if not m:
            continue
        fr = int(m.group(1))
        a, b, c = m.group(2), m.group(3), m.group(4)
        if c is None:
            sec = int(a) * 60 + int(b)
        else:
            sec = int(a) * 3600 + int(b) * 60 + int(c)
        times[fr] = sec
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prog_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--log", default="")
    ap.add_argument("--total_frames", type=int, default=85)
    ap.add_argument("--width", type=int, default=480,
                    help="渲染宽度；高度按原图比例算，fov 保持不变（等比缩小）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pose_from", default="",
                    help="从该快照取一个固定相机位姿，所有快照都用它渲染；"
                         "留空则每个快照用自己最后一个关键帧（视角会跟着走）")
    ap.add_argument("--pose_index", type=int, default=-1)
    ap.add_argument("--step", type=int, default=1,
                    help="每 step 个快照取一个，用于挑 12 个等间隔样本")
    ap.add_argument("--picks", default="",
                    help="显式指定要渲染的帧号，逗号分隔，如 36,40,44；留空则全取")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ns_names = sorted(d for d in os.listdir(args.prog_dir)
                      if os.path.isdir(os.path.join(args.prog_dir, d)))
    if args.step > 1:
        ns_names = ns_names[:: args.step]
    if args.picks:
        want = {int(x) for x in args.picks.split(",") if x.strip()}
        ns_names = [n for n in ns_names if int(n) in want]
    if args.limit:
        ns_names = ns_names[: args.limit]
    ftimes = parse_frame_times(args.log, args.total_frames)

    pose_kf = None
    if args.pose_from:
        pm = json.load(open(os.path.join(args.pose_from, "metadata.json")))
        pk = pm.get("keyframes") or []
        if not pk:
            raise SystemExit("pose_from 指定的快照还没有关键帧")
        pose_kf = pk[args.pose_index]
        print(f"[pose] 固定视角取自 {args.pose_from} 的 keyframes[{args.pose_index}]",
              flush=True)

    import cv2

    rows = []
    for name in ns_names:
        d = os.path.join(args.prog_dir, name)
        ply = os.path.join(d, "hierarchy.ply")
        if not os.path.isfile(ply):
            print(f"[skip] {name}: 没有 hierarchy.ply", flush=True)
            continue
        meta = json.load(open(os.path.join(d, "metadata.json")))
        kfs = meta.get("keyframes") or []
        if not kfs:
            print(f"[skip] {name}: 还没有关键帧", flush=True)
            continue

        W0 = int(meta["config"]["width"])
        H0 = int(meta["config"]["height"])
        kf = pose_kf if pose_kf is not None else kfs[-1]
        f = float(kf["f"])
        Rt = np.eye(4, dtype=np.float32)
        Rt[:4, :] = np.asarray(kf["Rt"], dtype=np.float32)

        Wr = args.width
        Hr = int(round(Wr * H0 / W0))
        fov_x = focal2fov(f, W0)
        fov_y = focal2fov(f, H0)
        vm = torch.from_numpy(Rt).cuda().transpose(0, 1)

        ns = SimpleNamespace(model_path=d, render_path="", out_dir=args.out_dir,
                             alignment_path="", framerate=30)
        scene = SceneModel.from_scene(d, ns)
        with torch.no_grad():
            r = scene.render(width=Wr, height=Hr, fov_x=fov_x, fov_y=fov_y,
                             view_matrix=vm, scaling_modifier=1.0)["render"]
        img = r.clamp(0, 1.0).mul(255).permute(1, 2, 0).byte().cpu().numpy()[:, :, ::-1]
        out_png = os.path.join(args.out_dir, f"{name}.png")
        cv2.imwrite(out_png, img)

        frame = int(name)
        rows.append({
            "snap": name,
            "frame": frame,
            "keyframes": len(kfs),
            "gaussians": ply_n_vertices(ply),
            "sec": ftimes.get(frame),
            "png": out_png,
        })
        print(f"[ok] {name}  frame={frame}  kf={len(kfs)}  "
              f"gauss={rows[-1]['gaussians']}  t={rows[-1]['sec']}s  -> {out_png}",
              flush=True)

        del scene
        torch.cuda.empty_cache()

    with open(os.path.join(args.out_dir, "summary.json"), "w") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    print(f"完成 {len(rows)} 张 -> {args.out_dir}/summary.json")


if __name__ == "__main__":
    main()
