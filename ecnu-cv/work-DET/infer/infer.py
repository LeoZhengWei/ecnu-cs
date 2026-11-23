# -*- coding: utf-8 -*-
"""检测推理与可视化。改写自样例 infer/infer.py，输出带框的图像。"""
import os
import sys
import json
import torch
import numpy as np
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset import VOC_CLASSES, VOCDetection
from model.net import build_model
from model.training import resolve_device

COLORS = [(255, 0, 0), (0, 180, 0), (0, 100, 255), (255, 160, 0), (180, 0, 200),
          (0, 200, 200), (120, 0, 0), (0, 120, 80), (200, 100, 0), (80, 80, 255)]


def load_model(ckpt_path, device, score_thresh=0.3):
    model = build_model(pretrained=False, score_thresh=score_thresh).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    return model


@torch.no_grad()
def detect_one(model, image, device):
    """返回 (boxes, scores, labels)，坐标还原到原图尺寸。"""
    w, h = image.size
    resized = TF.resize(image.convert("RGB"), (600, 800))
    tensor = TF.to_tensor(resized)
    out = model([tensor.to(device)])[0]
    boxes = out["boxes"].cpu().clone()
    if boxes.numel() > 0:
        boxes[:, 0::2] *= w / 800.0
        boxes[:, 1::2] *= h / 600.0
    return boxes, out["scores"].cpu(), out["labels"].cpu()


def draw_boxes(image, boxes, scores, labels):
    from model.net import COCO_TO_VOC
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for box, score, label in zip(boxes, scores, labels):
        vc = COCO_TO_VOC.get(int(label))
        if vc is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        c = vc - 1
        color = COLORS[c % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        name = VOC_CLASSES[c]
        draw.text((x1, max(0, y1 - 14)), "%s %.2f" % (name, score), fill=color)
    return img


def main():
    import argparse
    p = argparse.ArgumentParser(description="RetinaNet 检测推理")
    p.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "retinanet_best.pth"))
    p.add_argument("--data-root", default=os.path.join(ROOT, "datasets", "voc2007", "VOCdevkit"))
    p.add_argument("--out-dir", default=os.path.join(ROOT, "infer", "output_det"))
    p.add_argument("--num", type=int, default=16)
    p.add_argument("--score-thresh", type=float, default=0.5)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = resolve_device()
    model = load_model(args.ckpt, device, args.score_thresh)
    test_ds = VOCDetection(args.data_root, "test", augment=False)
    print("device", device, "images", min(args.num, len(test_ds)))
    summary = []
    for i in range(min(args.num, len(test_ds))):
        img_id = test_ds.ids[i]
        image = Image.open(os.path.join(test_ds.img_dir, img_id + ".jpg")).convert("RGB")
        boxes, scores, labels = detect_one(model, image, device)
        out = draw_boxes(image, boxes, scores, labels)
        out.save(os.path.join(args.out_dir, img_id + "_det.jpg"))
        from model.net import COCO_TO_VOC
        det = [{"cls": VOC_CLASSES[COCO_TO_VOC[int(l)] - 1], "score": round(float(s), 4),
                "box": [round(float(v), 1) for v in b]}
               for b, s, l in zip(boxes, scores, labels) if int(l) in COCO_TO_VOC]
        summary.append({"image": img_id, "detections": det})
        print("%s -> %d dets" % (img_id, len(det)))
    json.dump(summary, open(os.path.join(args.out_dir, "detections.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("DONE", args.out_dir)


if __name__ == "__main__":
    main()
