# -*- coding: utf-8 -*-
"""在 CamVid 测试集上汇总四种网络与两个基线的复原结果，重算指标与对比图。"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset import build_dataloaders
from evaluate.metrics import RestorationMeter, format_metrics
from model.net import build_generator, count_parameters
from model.training import center_mask, composite, fill_hole, resolve_device
import configs.config_loader as cfg_loader

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def to_hwc(image):
    return np.clip(image.transpose(1, 2, 0), 0, 1)


def nearest_paste(image, mask):
    """近邻粘贴：洞内每个像素取最近的洞外像素，作为不学习的基线。"""
    known = (mask[0, 0] < 0.5).cpu().numpy()
    _, inds = distance_transform_edt(~known, return_indices=True)
    out = image.clone()
    rows = torch.as_tensor(inds[0][~known], device=image.device)
    cols = torch.as_tensor(inds[1][~known], device=image.device)
    out[:, :, ~torch.as_tensor(known, device=image.device)] = image[:, :, rows, cols]
    return out


def main():
    args = cfg_loader.get_config()
    device = resolve_device()
    _train, _val, test_loader, test_pairs = build_dataloaders(args)
    out_dir = os.path.join(args.save_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    setups = [
        ("均值填充", None, None, 0),
        ("U-Net L2", "unet", "l2", args.base_unet),
        ("Context L2", "context", "l2", args.base_context),
        ("Context 联合", "context", "joint", args.base_context),
        ("U-Net 联合", "unet", "joint", args.base_unet),
        ("近邻粘贴", "nearest", None, 0),
    ]

    # 先把整张测试集按固定洞缓存下来，四个模型和两个基线共用同一批输入。
    cache = []
    with torch.no_grad():
        for image, name in test_loader:
            image = image.to(device)
            mask, _w = center_mask(image, args.hole, jitter=0, overlap=args.overlap)
            masked = fill_hole(image, mask)
            cache.append((image.cpu(), masked.cpu(), mask.cpu(), name))

    models = []
    for display, kind, loss, base in setups:
        if kind in (None, "nearest"):
            generator = None
            params = 0
            epoch = -1
        else:
            ckpt_path = os.path.join(args.save_dir, f"{kind}_{loss}", "best.pth")
            ckpt = torch_load(ckpt_path, device)
            generator = build_generator(kind, base_unet=args.base_unet,
                                        base_context=args.base_context).to(device)
            generator.load_state_dict(ckpt["generator"])
            generator.eval()
            params = count_parameters(generator)
            epoch = int(ckpt.get("epoch", -1))

        meter = RestorationMeter()
        panels = {}
        with torch.no_grad():
            for idx, (image, masked, mask, name) in enumerate(cache):
                im = image.to(device)
                mk = mask.to(device)
                ms = masked.to(device)
                if kind is None:
                    pred = ms
                elif kind == "nearest":
                    pred = nearest_paste(im, mk)
                else:
                    pred = generator(ms)
                done = composite(im, pred, mk)
                meter.update(pred.cpu().numpy(), im.cpu().numpy(), mk.cpu().numpy())
                if idx < 4:
                    panels[idx] = {
                        "orig": to_hwc(im[0].cpu().numpy()),
                        "mask": to_hwc(ms[0].cpu().numpy()),
                        "done": to_hwc(done[0].cpu().numpy()),
                    }
        stats = meter.compute()
        rec = {"display": display, "params": params, "epoch": epoch,
               "mae": stats["mae"], "mse": stats["mse"], "psnr": stats["psnr"],
               "ssim": stats["ssim"], "psnr_full": stats["psnr_full"],
               "count": stats["count"]}
        models.append(rec)
        print(format_metrics(stats), display)
        if display == "U-Net L2":
            first_panels = panels
        if display == "近邻粘贴":
            last_panels = panels

    # 汇总写盘
    summary = {"dataset": "CamVid", "hole": args.hole, "count": len(cache),
               "test_names": [n for _i, _m, _k, n in cache[:4]], "models": models}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 柱状图：PSNR 与 SSIM
    names = [m["display"] for m in models]
    psnr = [m["psnr"] for m in models]
    ssim = [m["ssim"] for m in models]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    bars0 = axes[0].bar(x, psnr, color=["#9aa0a6", "#4c8bf5", "#5fa8d3", "#7b68ee", "#ef8a62", "#f2c14e"])
    axes[0].set_ylabel("洞内 PSNR (dB)")
    axes[0].set_title("遮挡区域峰值信噪比")
    for b, v in zip(bars0, psnr):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.2f}", ha="center", fontsize=9)
    bars1 = axes[1].bar(x, ssim, color=["#9aa0a6", "#4c8bf5", "#5fa8d3", "#7b68ee", "#ef8a62", "#f2c14e"])
    axes[1].set_ylabel("整图 SSIM")
    axes[1].set_title("贴回原图后的结构相似度")
    axes[1].set_ylim(min(ssim) - 0.02, max(ssim) + 0.02)
    for b, v in zip(bars1, ssim):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "metrics_bar.png"), dpi=150)
    plt.close(fig)

    # 定性对比：原图 / 遮挡 / 四种网络 / 近邻粘贴
    cols = ["原图", "遮挡输入", "U-Net L2", "Context L2", "Context 联合", "U-Net 联合", "近邻粘贴"]
    # 重新跑四个生成器，取前 4 张补丁
    gens = {}
    for kind, loss in [("unet", "l2"), ("context", "l2"), ("context", "joint"), ("unet", "joint")]:
        ckpt = torch_load(os.path.join(args.save_dir, f"{kind}_{loss}", "best.pth"), device)
        g = build_generator(kind, base_unet=args.base_unet, base_context=args.base_context).to(device)
        g.load_state_dict(ckpt["generator"])
        g.eval()
        gens[(kind, loss)] = g
    order = [("orig", None), ("mask", None), ("unet", "l2"), ("context", "l2"),
             ("context", "joint"), ("unet", "joint"), ("nearest", None)]
    fig, axes = plt.subplots(4, 7, figsize=(15, 8))
    with torch.no_grad():
        for r in range(4):
            image, masked, mask, _name = cache[r]
            im = image.to(device)
            mk = mask.to(device)
            ms = masked.to(device)
            patches = {
                "orig": to_hwc(im[0].cpu().numpy()),
                "mask": to_hwc(ms[0].cpu().numpy()),
                "nearest": to_hwc(composite(im, nearest_paste(im, mk), mk)[0].cpu().numpy()),
            }
            for (kind, loss), g in gens.items():
                done = composite(im, g(ms), mk)
                patches[kind] = to_hwc(done[0].cpu().numpy())
            for c, (kind, loss) in enumerate(order):
                key = "orig" if kind == "orig" else ("mask" if kind == "mask" else ("nearest" if kind == "nearest" else kind))
                ax = axes[r, c]
                ax.imshow(patches[key])
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(cols[c], fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "qualitative_compare.png"), dpi=140)
    plt.close(fig)

    print("SUMMARY_OK", out_dir)
    for m in models:
        print("%-12s PSNR=%.2f SSIM=%.4f epoch=%d" % (m["display"], m["psnr"], m["ssim"], m["epoch"]))


if __name__ == "__main__":
    main()
