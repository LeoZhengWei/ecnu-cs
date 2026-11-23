# -*- coding: utf-8 -*-
"""训练循环。改写自样例 model/training.py。

使用 torchvision 官方 RetinaNet 的前向接口：训练模式下 model(images, targets)
直接返回包含分类损失与回归损失的字典；评测调用 evaluate.compute_map。
"""
import os
import time
import torch
from torch.utils.data import DataLoader

# 避免 DataLoader 多进程累积打开过多文件描述符
torch.multiprocessing.set_sharing_strategy("file_system")

from dataset import VOCDetection, collate_fn
from evaluate import compute_map


def resolve_device():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            if "5090" in torch.cuda.get_device_name(i):
                return torch.device("cuda:%d" % i)
        return torch.device("cuda:0")
    return torch.device("cpu")


class Trainer(object):
    def __init__(self, model, train_dataset, test_dataset, args):
        self.args = args
        self.model = model
        self.device = resolve_device()
        self.model.to(self.device)
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                       shuffle=True, collate_fn=collate_fn, num_workers=2,
                                       persistent_workers=True)
        self.test_loader = DataLoader(test_dataset, batch_size=1,
                                      shuffle=False, collate_fn=collate_fn, num_workers=0)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[int(args.epochs * 0.6), int(args.epochs * 0.85)], gamma=0.1)
        self.epochs = args.epochs
        self.best_map = 0.0
        os.makedirs(args.ckpt_dir, exist_ok=True)
        print("Using", self.device, "batches/epoch", len(self.train_loader), flush=True)

    def train_one_epoch(self, epoch):
        self.model.train()
        t0 = time.time(); running = {"cls": 0.0, "reg": 0.0, "total": 0.0}; steps = 0
        for images, targets in self.train_loader:
            images = [im.to(self.device) for im in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            loss_dict = self.model(images, targets)
            loss = sum(v for v in loss_dict.values())
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            running["total"] += loss.item()
            running["cls"] += float(loss_dict.get("classification", 0))
            running["reg"] += float(loss_dict.get("box_regression", 0))
            steps += 1
        self.scheduler.step()
        n = max(steps, 1)
        print("epoch %02d loss=%.4f (cls=%.4f reg=%.4f) lr=%.2e %.1fs" % (
            epoch, running["total"] / n, running["cls"] / n, running["reg"] / n,
            self.optimizer.param_groups[0]["lr"], time.time() - t0), flush=True)

    @torch.no_grad()
    def evaluate(self, epoch):
        results = compute_map(self.model, self.test_loader, self.device,
                              score_thresh=self.args.score_thresh)
        mAP = results["mAP"]
        print("epoch %02d  mAP@0.5=%.4f  (best %.4f)" % (epoch, mAP, self.best_map), flush=True)
        if mAP >= self.best_map:
            self.best_map = max(self.best_map, mAP)
            torch.save({
                "model": self.model.state_dict(),
                "mAP": mAP, "epoch": epoch,
                "ap_per_class": results["ap_per_class"],
            }, os.path.join(self.args.ckpt_dir, "retinanet_best.pth"))
        return results

    def train_model(self):
        history = []
        for epoch in range(1, self.epochs + 1):
            self.train_one_epoch(epoch)
            if epoch % self.args.eval_every == 0 or epoch == self.epochs:
                results = self.evaluate(epoch)
                history.append({"epoch": epoch, "mAP": results["mAP"],
                                "ap_per_class": results["ap_per_class"],
                                "n_gt": results.get("n_gt", {})})
        print("BEST_mAP=%.4f" % self.best_map, flush=True)
        return history
