"""画出测试照片：原图、挖空后的输入、这个 GAN 补回去的结果。"""

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

from dataset import build_dataloaders
from model.net import build_generator
from model.training import center_mask, composite, fill_hole, resolve_device
import configs.config_loader as cfg_loader

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
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


def main():
    args = cfg_loader.get_config()
    device = resolve_device()
    _train, _val, test_loader, _pairs = build_dataloaders(args)
    ckpt_path = os.path.join(args.save_dir, "context_joint", "best.pth")
    ckpt = torch_load(ckpt_path, device)
    generator = build_generator("context", base_context=args.base_context).to(device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()

    originals, masked_list, restored, names = [], [], [], []
    with torch.no_grad():
        for image, name in test_loader:
            image = image.to(device)
            mask, _weight = center_mask(image, args.hole, jitter=0, overlap=args.overlap)
            masked = fill_hole(image, mask)
            pred = generator(masked)
            done = composite(image, pred, mask)
            for i in range(image.shape[0]):
                originals.append(image[i].cpu().numpy())
                masked_list.append(masked[i].cpu().numpy())
                restored.append(done[i].cpu().numpy())
                names.append(name[i])

    # 数据集里有些照片带大块灰边，挑中间有窗户、灰边少的几张来看。
    scores = []
    for image in originals:
        h, w = image.shape[1], image.shape[2]
        y0, x0 = (h - args.hole) // 2, (w - args.hole) // 2
        patch = image[:, y0:y0 + args.hole, x0:x0 + args.hole]
        chroma = image.max(axis=0) - image.min(axis=0)
        gray = float(((chroma < 0.04) & (image.mean(axis=0) > 0.35) & (image.mean(axis=0) < 0.75)).mean())
        scores.append(float(patch.std()) - 2.0 * gray)
    order = list(np.argsort(scores)[::-1])
    pick = []
    for index in order:
        if len(pick) >= 4:
            break
        if not pick or abs(int(index) - pick[-1]) > 2:
            pick.append(int(index))

    fig, axes = plt.subplots(len(pick), 3, figsize=(7.2, 9.2))
    titles = ["原图", "挖掉中间", "补回去"]
    for row, index in enumerate(pick):
        panels = [originals[index], masked_list[index], restored[index]]
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(to_hwc(panel))
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col], fontsize=12)
            if col == 0:
                ax.set_ylabel(names[index].replace(".jpg", ""), fontsize=8)
    fig.tight_layout()
    out_dir = os.path.join(args.save_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test_restore.png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    with open(os.path.join(out_dir, "picked.json"), "w", encoding="utf-8") as f:
        json.dump({"files": [names[i] for i in pick]}, f, ensure_ascii=False, indent=2)
    print(out_path)
    print([names[i] for i in pick])


if __name__ == "__main__":
    main()
