"""在测试集上对比已训练模型，写出指标并画定性结果图。"""

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

from dataset import CLASS_COLORS, CLASS_NAMES_ZH, build_dataloaders, colorize, make_splits
from evaluate.metrics import ConfusionMeter, format_metrics
from model.net import build_model, count_parameters
from model.training import resolve_device

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

MODELS = [
    ("unet", "残差 U-Net"),
    ("fcn", "无跳跃连接"),
    ("segnet", "SegNet"),
    ("deeplab", "DeepLabV3+"),
    ("attunet", "Attention U-Net"),
]


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(name, ckpt_path, num_classes, base, device):
    ckpt = torch_load(ckpt_path, device)
    model = build_model(name, 3, num_classes, base=base)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, count_parameters(model)


@torch.no_grad()
def predict_loader(model, loader, device):
    preds = []
    for img, _mask in loader:
        img = img.to(device, non_blocking=True)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(img)
        else:
            logits = model(img)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def collect_split(loader):
    images, masks = [], []
    for img, mask in loader:
        images.append(img.numpy())
        masks.append(mask.numpy())
    return np.concatenate(images, axis=0), np.concatenate(masks, axis=0)


def evaluate_predictions(preds, masks, num_classes, ignore_index):
    meter = ConfusionMeter(num_classes, ignore_index)
    meter.update(preds, masks)
    return meter.compute()


def stats_to_json(stats, params, display):
    per_class = {}
    for i, name in enumerate(CLASS_NAMES_ZH):
        def cell(arr):
            value = float(arr[i])
            return None if np.isnan(value) else value

        per_class[name] = {
            "iou": cell(stats["iou"]),
            "precision": cell(stats["precision"]),
            "recall": cell(stats["recall"]),
            "dice": cell(stats["dice"]),
            "support": int(stats["support"][i]),
        }
    return {
        "display": display,
        "params": int(params),
        "pixel_acc": stats["pixel_acc"],
        "mean_acc": stats["mean_acc"],
        "miou": stats["miou"],
        "fwiou": stats["fwiou"],
        "mdice": stats["mdice"],
        "per_class": per_class,
    }


def draw_qualitative(images, masks, predictions, indices, names, out_path):
    columns = ["原图", "标注"] + [title for _name, title in MODELS]
    nrows = len(indices)
    ncols = len(columns)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.15, nrows * 2.15 + 0.75))
    for row, index in enumerate(indices):
        panels = [np.transpose(images[index], (1, 2, 0)), colorize(masks[index])]
        panels.extend(colorize(predictions[name][index]) for name, _title in MODELS)
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(np.clip(panel, 0, 1) if panel.dtype != np.uint8 else panel)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(columns[col], fontsize=11, pad=4)
            if col == 0:
                ax.set_ylabel(names[index], fontsize=8)
    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=tuple(color / 255), markersize=10, label=name)
        for name, color in zip(CLASS_NAMES_ZH, CLASS_COLORS)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=11, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def draw_metrics(summary, out_path):
    names = [item["display"] for item in summary]
    miou = [item["miou"] for item in summary]
    pa = [item["pixel_acc"] for item in summary]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    bars_m = ax.bar(x - width / 2, miou, width, label="mIoU", color="#3d5a80")
    bars_p = ax.bar(x + width / 2, pa, width, label="像素准确率", color="#98c1d9")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("分数")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bars in (bars_m, bars_p):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    os.chdir(ROOT)
    import configs.config_loader as cfg_loader

    args = cfg_loader.get_config()
    _train_loader, _val_loader, test_loader, _weights, _counts = build_dataloaders(args, with_weights=False)
    images, masks = collect_split(test_loader)
    device = resolve_device()
    print(f"device={device} test={len(masks)}")
    predictions = {}
    summary = []
    for name, display in MODELS:
        ckpt = os.path.join(args.save_dir, name, "best.pth")
        model, params = load_model(name, ckpt, args.num_classes, args.base_channels, device)
        pred = predict_loader(model, test_loader, device)
        predictions[name] = pred
        stats = evaluate_predictions(pred, masks, args.num_classes, args.ignore_index)
        print(f"===== {display} params={params} =====")
        print(format_metrics(stats, CLASS_NAMES_ZH))
        summary.append(stats_to_json(stats, params, display))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _base, _train_pairs, _val_pairs, test_pairs = make_splits(args.data_root, seed=args.seed)
    names = [os.path.splitext(os.path.basename(path))[0] for path, _label in test_pairs]
    # 测试集是同一段视频。隔开选取，覆盖车辆、路口、近处骑行者和弯道。
    indices = [0, 36, 60, 84]
    fig_dir = os.path.join(args.save_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    qual_path = os.path.join(fig_dir, "qualitative_compare.png")
    bar_path = os.path.join(fig_dir, "metrics_bar.png")
    draw_qualitative(images, masks, predictions, indices, names, qual_path)
    draw_metrics(summary, bar_path)
    payload = {
        "test_indices": indices,
        "test_names": [names[i] for i in indices],
        "models": summary,
    }
    out_json = os.path.join(fig_dir, "summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"figure={qual_path}")
    print(f"metrics={bar_path}")
    print(f"summary={out_json}")
    print("indices", indices)


if __name__ == "__main__":
    main()
