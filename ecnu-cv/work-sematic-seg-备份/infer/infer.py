import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset import CLASS_NAMES_ZH, colorize
from model.net import build_model
from model.training import resolve_device


def load_model(ckpt_path, device):
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
    else:
        model = build_model(
            ckpt.get("model", "unet"),
            in_channels=ckpt.get("in_channels", 3),
            out_channels=ckpt.get("out_channels", 11),
            base=ckpt.get("base_channels", 64),
        )
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    height = ckpt.get("img_height", 256) if isinstance(ckpt, dict) else 256
    width = ckpt.get("img_width", 320) if isinstance(ckpt, dict) else 320
    return model, (height, width)


def predict(model, image_path, device, img_size):
    image = Image.open(image_path).convert("RGB")
    original = np.array(image)
    resized = TF.resize(image, img_size, interpolation=TF.InterpolationMode.BILINEAR)
    tensor = TF.to_tensor(resized).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    return original, pred


def save_prediction(original, pred, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    color = colorize(pred)
    Image.fromarray(color).save(save_path)
    overlay_base = Image.fromarray(original).resize((pred.shape[1], pred.shape[0]), Image.BILINEAR)
    base = np.array(overlay_base).astype(np.float32)
    blended = (0.55 * base + 0.45 * color.astype(np.float32)).astype(np.uint8)
    overlay_path = os.path.splitext(save_path)[0] + "_overlay.png"
    Image.fromarray(blended).save(overlay_path)
    return overlay_path


def main():
    parser = argparse.ArgumentParser(description="CamVid 语义分割推理")
    parser.add_argument("--ckpt", default=os.path.join(ROOT, "runs", "unet", "best.pth"))
    parser.add_argument("--image", default=None)
    parser.add_argument("--image-dir", default=os.path.join(ROOT, "data", "camvid", "images"))
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "infer", "output"))
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    device = resolve_device()
    model, img_size = load_model(args.ckpt, device)
    if args.image:
        paths = [args.image]
    else:
        names = sorted(
            name for name in os.listdir(args.image_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        )[: args.limit]
        paths = [os.path.join(args.image_dir, name) for name in names]
    print(f"device={device} images={len(paths)} ckpt={args.ckpt} classes={CLASS_NAMES_ZH}")
    for image_path in paths:
        original, pred = predict(model, image_path, device, img_size)
        name = os.path.splitext(os.path.basename(image_path))[0] + ".png"
        save_path = os.path.join(args.out_dir, name)
        overlay_path = save_prediction(original, pred, save_path)
        print(f"{os.path.basename(image_path)} -> {save_path} overlay {overlay_path}")


if __name__ == "__main__":
    main()
