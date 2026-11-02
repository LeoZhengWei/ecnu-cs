"""比较 U-Net 和 FCN 的测试指标，并保存训练曲线和预测可视化。"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import CLASS_COLORS, CLASS_NAMES, CLASS_NAMES_ZH, build_dataloaders, colorize
from model.net import build_model
from model.training import resolve_device
import configs.config_loader as cfg_loader


def load_checkpoint(path, device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    model = build_model(
        ckpt["model"],
        in_channels=ckpt.get("in_channels", 3),
        out_channels=ckpt["out_channels"],
        base=ckpt.get("base_channels", 64),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def plot_history(run_root, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in ("unet", "fcn"):
        path = os.path.join(run_root, name, "history.json")
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)
        epochs = [item["epoch"] for item in history]
        axes[0].plot(epochs, [item["train_loss"] for item in history], label=f"{name} train")
        if "val_loss" in history[0]:
            axes[0].plot(epochs, [item["val_loss"] for item in history], linestyle="--", label=f"{name} val")
        axes[1].plot(epochs, [item.get("val_miou", 0) for item in history], label=name)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].legend()
    axes[1].set_title("Validation mIoU")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"curve -> {out_path}")


def save_qualitative(models, loader, device, out_path, max_images=4):
    collected = []
    for img, mask in loader:
        img = img.to(device)
        with torch.no_grad():
            preds = {name: model(img).argmax(dim=1).cpu().numpy() for name, model in models.items()}
        for b in range(img.shape[0]):
            collected.append(
                (
                    (img[b].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8),
                    mask[b].numpy(),
                    {name: preds[name][b] for name in preds},
                )
            )
    if len(collected) > max_images:
        picks = [round(i * (len(collected) - 1) / (max_images - 1)) for i in range(max_images)]
        collected = [collected[i] for i in picks]

    names = list(models)
    cols = 2 + len(names)
    fig, axes = plt.subplots(len(collected), cols, figsize=(3.2 * cols, 3 * len(collected)))
    if len(collected) == 1:
        axes = np.expand_dims(axes, 0)
    titles = ["Image", "Ground truth"] + names
    for row, (image, gt, pred_map) in enumerate(collected):
        panels = [image, colorize(gt)] + [colorize(pred_map[name]) for name in names]
        for col, panel in enumerate(panels):
            axes[row, col].imshow(panel)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(titles[col])
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"qualitative -> {out_path}")


def write_report_table(run_root, out_path):
    rows = []
    per_class = {}
    for name in ("unet", "fcn"):
        path = os.path.join(run_root, name, "test_metrics.json")
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        rows.append(stats | {"model": name})
        per_class[name] = stats["per_class"]
    lines = [
        "| 模型 | PA | mPA | mIoU | FWIoU | mDice |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stats in rows:
        lines.append(
            f"| {stats['model']} | {stats['pixel_acc']:.4f} | {stats['mean_acc']:.4f} | "
            f"{stats['miou']:.4f} | {stats['fwiou']:.4f} | {stats['mdice']:.4f} |"
        )
    lines += [
        "",
        "| 类别 | U-Net IoU | FCN IoU | U-Net Dice | FCN Dice |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for eng, zh in zip(CLASS_NAMES, CLASS_NAMES_ZH):
        unet_iou = per_class.get("unet", {}).get(eng, {}).get("iou")
        fcn_iou = per_class.get("fcn", {}).get(eng, {}).get("iou")
        unet_dice = per_class.get("unet", {}).get(eng, {}).get("dice")
        fcn_dice = per_class.get("fcn", {}).get(eng, {}).get("dice")

        def fmt(value):
            return "nan" if value is None else f"{value:.4f}"

        lines.append(f"| {zh} ({eng}) | {fmt(unet_iou)} | {fmt(fcn_iou)} | {fmt(unet_dice)} | {fmt(fcn_dice)} |")
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


def main():
    args = cfg_loader.get_config()
    device = resolve_device()
    fig_dir = os.path.join(args.save_dir, "figures")
    plot_history(args.save_dir, os.path.join(fig_dir, "curves.png"))

    models = {}
    for name in ("unet", "fcn"):
        path = os.path.join(args.save_dir, name, "best.pth")
        if os.path.isfile(path):
            model, _ = load_checkpoint(path, device)
            models[name] = model
    if models:
        args.num_workers = 0
        _, _, test_loader, _, _ = build_dataloaders(args, with_weights=False)
        save_qualitative(models, test_loader, device, os.path.join(fig_dir, "qualitative.png"))
    write_report_table(args.save_dir, os.path.join(fig_dir, "metrics_table.md"))
    legend = os.path.join(fig_dir, "colormap.png")
    fig, ax = plt.subplots(figsize=(4.2, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(len(CLASS_NAMES), -1)
    ax.axis("off")
    for i, (color, eng) in enumerate(zip(CLASS_COLORS, CLASS_NAMES)):
        ax.add_patch(plt.Rectangle((0.2, i - 0.35), 1.6, 0.7, color=color / 255.0))
        ax.text(2.2, i, eng, va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(legend, dpi=140)
    plt.close(fig)
    print(f"colormap -> {legend}")


if __name__ == "__main__":
    main()
