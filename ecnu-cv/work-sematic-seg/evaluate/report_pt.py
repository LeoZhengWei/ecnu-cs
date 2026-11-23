# -*- coding: utf-8 -*-
"""只评测 EfficientNet-B0 的 ImageNet 预训练变体，并画一张与从头训练版的对照图。

输出：
  runs/figures/summary_pt.json       逐类与总体指标
  runs/figures/efficientnet_variants.png  原图/标注/从头训练/预训练 四列对照
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset import CLASS_COLORS, CLASS_NAMES_ZH, build_dataloaders, colorize
from evaluate.compare import (collect_split, evaluate_predictions,
                              predict_loader, stats_to_json, torch_load)
from model.net import build_model, count_parameters
from model.training import resolve_device

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

VARIANTS = [
    ("efficientnet", "EfficientNet-B0 从头训练"),
    ("efficientnet_pt", "EfficientNet-B0 预训练微调"),
]


def load(name, ckpt_path, num_classes, base, device):
    ckpt = torch_load(ckpt_path, device)
    model = build_model(name, 3, num_classes, base=base)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, count_parameters(model)


def draw(images, masks, predictions, names, indices, out_path):
    columns = ["原图", "标注", "EffNet-B0 从头", "EffNet-B0 预训练"]
    nrows, ncols = len(indices), len(columns)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.7, nrows * 2.7 + 0.8))
    for row, index in enumerate(indices):
        panels = [np.transpose(images[index], (1, 2, 0)), colorize(masks[index])]
        panels.extend(colorize(predictions[name][index]) for name, _t in VARIANTS)
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(np.clip(panel, 0, 1) if panel.dtype != np.uint8 else panel)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(columns[col], fontsize=12, pad=5)
            if col == 0:
                ax.set_ylabel(names[index], fontsize=9)
    handles = [
        plt.Line2D([0], [0], marker="s", color="none",
                   markerfacecolor=tuple(c / 255), markersize=10, label=n)
        for n, c in zip(CLASS_NAMES_ZH, CLASS_COLORS)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=11, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    os.chdir(ROOT)
    import configs.config_loader as cfg_loader

    args = cfg_loader.get_config()
    _tr, _va, test_loader, _w, _c = build_dataloaders(args, with_weights=False)
    images, masks = collect_split(test_loader)
    device = resolve_device()
    print("device=%s test=%d" % (device, len(masks)))

    predictions, summary = {}, []
    for name, display in VARIANTS:
        ckpt = os.path.join(args.save_dir, name, "best.pth")
        model, params = load(name, ckpt, args.num_classes, args.base_channels, device)
        pred = predict_loader(model, test_loader, device)
        predictions[name] = pred
        stats = evaluate_predictions(pred, masks, args.num_classes, args.ignore_index)
        item = stats_to_json(stats, params, display)
        print("===== %s params=%d miou=%.4f PA=%.4f mPA=%.4f FWIoU=%.4f mDice=%.4f" %
              (display, params, item["miou"], item["pixel_acc"], item["mean_acc"],
               item["fwiou"], item["mdice"]))
        summary.append(item)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    from dataset import make_splits
    _b, _trp, _vap, test_pairs = make_splits(args.data_root, seed=args.seed)
    names = [os.path.splitext(os.path.basename(p))[0] for p, _l in test_pairs]
    fig_dir = os.path.join(args.save_dir, "figures")
    with open(os.path.join(fig_dir, "summary_pt.json"), "w", encoding="utf-8") as f:
        json.dump({"models": summary}, f, ensure_ascii=False, indent=2)
    draw(images, masks, predictions, names, [0, 36, 60, 84],
         os.path.join(fig_dir, "efficientnet_variants.png"))
    print("WROTE summary_pt.json efficientnet_variants.png")


if __name__ == "__main__":
    main()
