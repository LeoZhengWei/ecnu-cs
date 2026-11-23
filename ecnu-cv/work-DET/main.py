# -*- coding: utf-8 -*-
"""RetinaNet 训练入口。改写自课程样例 main.py，保持同样的框架组织方式。"""
import os
import sys
import json
import argparse
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset import VOCDetection
from model.net import build_model
from model.training import Trainer, resolve_device


def get_args():
    p = argparse.ArgumentParser(description="RetinaNet 目标检测（PASCAL VOC 2007）")
    p.add_argument("--data-root", default=os.path.join(ROOT, "datasets", "voc2007", "VOCdevkit"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-reg", type=float, default=1.0)
    p.add_argument("--score-thresh", type=float, default=0.05)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--ckpt-dir", default=os.path.join(ROOT, "checkpoints"))
    p.add_argument("--out-json", default=os.path.join(ROOT, "metrics.json"))
    p.add_argument("--resume", default="")
    return p.parse_args()


def main():
    args = get_args()
    torch.manual_seed(42)
    print("args:", vars(args), flush=True)
    train_ds = VOCDetection(args.data_root, "trainval", augment=True)
    test_ds = VOCDetection(args.data_root, "test", augment=False)
    print("train=%d test=%d" % (len(train_ds), len(test_ds)), flush=True)
    model = build_model(pretrained=True, score_thresh=args.score_thresh)
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        print("resumed from", args.resume, flush=True)
    trainer = Trainer(model, train_ds, test_ds, args)
    history = trainer.train_model()
    # 保存指标
    out = {"best_map": trainer.best_map, "history": history, "args": vars(args)}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print("SAVED", args.out_json, flush=True)


if __name__ == "__main__":
    main()
