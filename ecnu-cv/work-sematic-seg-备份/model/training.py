from __future__ import division

import json
import os
import random
import time

import numpy as np
import torch

from metrics import ConfusionMeter, format_metrics
from dataset import CLASS_NAMES


def resolve_device():
    """按当前机器选择训练设备：5090、3050（cuda:0）或 Mac 的 MPS。"""
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            if "5090" in torch.cuda.get_device_name(index):
                return torch.device(f"cuda:{index}")
        for index in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(index)
            if "3050" in name:
                return torch.device("cuda:0")
        return torch.device("cuda:0")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer(object):
    def __init__(self, train_loader, model, opt, val_loader=None, class_weights=None):
        self.args = opt
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.epochs = opt.epochs
        self.device = resolve_device()
        set_seed(opt.seed)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            device_name = torch.cuda.get_device_name(self.device)
            torch.backends.cudnn.benchmark = True
        else:
            device_name = self.device.type
        self.model.to(self.device)
        print(f"Using {self.device}: {device_name}")

        weight = None
        if class_weights is not None:
            weight = class_weights.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=opt.ignore_index)
        self.optimizer = torch.optim.AdamW(
            lr=opt.lr, weight_decay=opt.weight_decay, params=model.parameters()
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(opt.epochs, 1))
        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16
        self.run_dir = os.path.join(opt.save_dir, opt.model.lower())
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs("pth", exist_ok=True)
        self.history = []
        self.best_miou = -1.0
        self.best_path = os.path.join(self.run_dir, "best.pth")

    def _step_loss(self, img, mask):
        if self.use_amp:
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                output = self.model(img)
                loss = self.criterion(output, mask)
            return output, loss
        output = self.model(img)
        return output, self.criterion(output, mask)

    def train_model(self):
        for epoch in range(self.epochs):
            self.model.train()
            losses = []
            start = time.time()
            for i, (img, mask) in enumerate(self.train_loader):
                img = img.to(self.device, non_blocking=True)
                mask = mask.long().to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                _, loss = self._step_loss(img, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                losses.append(loss.item())
                if i % 20 == 0:
                    print(f"epoch {epoch} step {i} train_loss {loss.item():.4f}")
            train_loss = float(sum(losses) / max(len(losses), 1))
            record = {"epoch": epoch, "train_loss": train_loss, "lr": self.optimizer.param_groups[0]["lr"]}
            self.scheduler.step()
            message = f"epoch {epoch} train_loss {train_loss:.4f} time {time.time() - start:.1f}s"
            if self.val_loader is not None:
                val_loss, stats = self.evaluate(self.val_loader, split="val")
                record.update(
                    {
                        "val_loss": val_loss,
                        "val_miou": stats["miou"],
                        "val_pixel_acc": stats["pixel_acc"],
                        "val_mdice": stats["mdice"],
                    }
                )
                message += (
                    f" val_loss {val_loss:.4f} val_mIoU {stats['miou']:.4f} val_PA {stats['pixel_acc']:.4f}"
                )
                if stats["miou"] > self.best_miou:
                    self.best_miou = stats["miou"]
                    self._save(self.best_path, epoch, stats["miou"])
                    message += "  *best"
            print(message)
            self.history.append(record)
            self._write_history()

        last_path = os.path.join(self.run_dir, "last.pth")
        self._save(last_path, self.epochs - 1, self.best_miou)
        legacy = os.path.join("pth", f"{self.args.model}-epochs{self.epochs}.pth")
        self._save(legacy, self.epochs - 1, self.best_miou)
        print(f"best val mIoU {self.best_miou:.4f} -> {self.best_path}")
        return self.history

    @torch.no_grad()
    def evaluate(self, loader, split="test"):
        self.model.eval()
        meter = ConfusionMeter(self.args.num_classes, self.args.ignore_index)
        losses = []
        for img, mask in loader:
            img = img.to(self.device, non_blocking=True)
            mask = mask.long().to(self.device, non_blocking=True)
            output, loss = self._step_loss(img, mask)
            losses.append(loss.item())
            pred = output.argmax(dim=1)
            meter.update(pred.cpu().numpy(), mask.cpu().numpy())
        stats = meter.compute()
        mean_loss = float(sum(losses) / max(len(losses), 1))
        if split != "val":
            text = format_metrics(stats, CLASS_NAMES)
            print(f"===== {split} =====\n{text}")
            payload = {
                "split": split,
                "loss": mean_loss,
                "pixel_acc": stats["pixel_acc"],
                "mean_acc": stats["mean_acc"],
                "miou": stats["miou"],
                "fwiou": stats["fwiou"],
                "mdice": stats["mdice"],
                "per_class": {
                    name: {
                        "iou": None if np.isnan(stats["iou"][i]) else float(stats["iou"][i]),
                        "precision": None if np.isnan(stats["precision"][i]) else float(stats["precision"][i]),
                        "recall": None if np.isnan(stats["recall"][i]) else float(stats["recall"][i]),
                        "dice": None if np.isnan(stats["dice"][i]) else float(stats["dice"][i]),
                        "support": int(stats["support"][i]),
                    }
                    for i, name in enumerate(CLASS_NAMES)
                },
            }
            out_path = os.path.join(self.run_dir, f"{split}_metrics.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"metrics -> {out_path}")
        self.model.train()
        return mean_loss, stats

    def _save(self, path, epoch, miou):
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "epoch": epoch,
                "miou": miou,
                "model": self.args.model,
                "in_channels": 3,
                "out_channels": self.args.num_classes,
                "base_channels": self.args.base_channels,
                "img_height": self.args.img_height,
                "img_width": self.args.img_width,
            },
            path,
        )

    def _write_history(self):
        path = os.path.join(self.run_dir, "history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
