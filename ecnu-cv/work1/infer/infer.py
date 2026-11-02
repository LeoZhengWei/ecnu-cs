import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.net import UNet
from model.training import resolve_device


def load_model(ckpt_path, device):
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
    else:
        model = UNet(in_channels=1, out_channels=1)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


def infer(model, image_path, device):
    """读一张灰度图，返回与原图同大小的前景概率图。"""
    image = Image.open(image_path).convert("L")
    tensor = transforms.ToTensor()(image).unsqueeze(0)
    _, _, height, width = tensor.shape
    pad_h = (16 - height % 16) % 16
    pad_w = (16 - width % 16) % 16
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h))

    tensor = tensor.to(device)
    with torch.no_grad():
        output = model(tensor)

    prob = output[0, 0, :height, :width].detach().cpu().numpy()
    return np.array(image), prob


def save_mask(image, prob, save_path, threshold):
    mask = (prob >= threshold).astype(np.uint8) * 255
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(mask).save(save_path)

    overlay = np.stack([image, image, image], axis=-1)
    color = overlay.copy()
    color[mask > 0] = (255, 40, 40)
    blended = (0.65 * overlay + 0.35 * color).astype(np.uint8)
    overlay_path = os.path.splitext(save_path)[0] + "_overlay.png"
    Image.fromarray(blended).save(overlay_path)
    return mask, overlay_path


def collect_images(image, image_dir):
    if image:
        return [image]
    names = sorted(
        name for name in os.listdir(image_dir)
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    )
    return [os.path.join(image_dir, name) for name in names]


def main():
    parser = argparse.ArgumentParser(description="U-Net 肾脏分割推理")
    parser.add_argument("--ckpt", default=os.path.join(ROOT, "pth", "ResUnet-epochs200.pth"))
    parser.add_argument("--image", default=None, help="单张待预测图片")
    parser.add_argument("--image-dir", default=os.path.join(ROOT, "data", "val", "img"))
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "infer", "output"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = resolve_device()
    model = load_model(args.ckpt, device)
    paths = collect_images(args.image, args.image_dir)
    print(f"device={device}  images={len(paths)}  ckpt={args.ckpt}")

    for image_path in paths:
        image, prob = infer(model, image_path, device)
        name = os.path.splitext(os.path.basename(image_path))[0] + ".png"
        save_path = os.path.join(args.out_dir, name)
        mask, overlay_path = save_mask(image, prob, save_path, args.threshold)
        foreground = int((mask > 0).sum())
        print(f"{os.path.basename(image_path)}  prob={prob.shape}  foreground={foreground}  -> {save_path}")
        print(f"overlay -> {overlay_path}")


if __name__ == "__main__":
    main()
