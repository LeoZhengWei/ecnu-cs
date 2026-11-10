import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from evaluate.metrics import RestorationMeter, format_metrics
from model.net import Discriminator, build_generator, count_parameters


def resolve_device():
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            if "5090" in torch.cuda.get_device_name(index):
                return torch.device(f"cuda:{index}")
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


def run_name(model, loss):
    return f"{model.lower()}_{loss.lower()}"


def center_mask(batch, hole, jitter, overlap):
    """返回洞的掩膜（1 为缺失）和重建权重。重叠带是洞外扩 overlap 像素，权重为 10。"""
    _, _, height, width = batch.shape
    mask = torch.zeros(batch.shape[0], 1, height, width, device=batch.device)
    y_mid = (height - hole) // 2
    x_mid = (width - hole) // 2
    for index in range(batch.shape[0]):
        dy = random.randint(-jitter, jitter) if jitter else 0
        dx = random.randint(-jitter, jitter) if jitter else 0
        y0 = min(max(overlap + 1, y_mid + dy), height - hole - overlap - 1)
        x0 = min(max(overlap + 1, x_mid + dx), width - hole - overlap - 1)
        mask[index, :, y0:y0 + hole, x0:x0 + hole] = 1
    kernel = overlap * 2 + 1
    dilated = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=overlap)
    band = (dilated - mask).clamp(0, 1)
    # 接缝那一圈权重大一些，把补丁粘住；洞里面仍然占损失的大头。
    weight = mask + 4.0 * band
    return mask, weight


def fill_hole(image, mask):
    known = 1.0 - mask
    area = known.sum(dim=(2, 3), keepdim=True).clamp(min=1)
    mean = (image * known).sum(dim=(2, 3), keepdim=True) / area
    return image * known + mean * mask


def composite(image, pred, mask):
    return image * (1.0 - mask) + pred * mask


class Trainer(object):
    def __init__(self, train_loader, val_loader, args):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = resolve_device()
        set_seed(args.seed)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.backends.cudnn.benchmark = True
            device_name = torch.cuda.get_device_name(self.device)
        else:
            device_name = self.device.type
        print(f"Using {self.device}: {device_name}")

        self.generator = build_generator(args.model, args.base_unet, args.base_context).to(self.device)
        self.use_adv = args.loss in ("joint", "adv")
        self.use_rec = args.loss in ("joint", "l2")
        self.warmup_epochs = int(getattr(args, "warmup_epochs", 0))
        self.discriminator = Discriminator().to(self.device) if self.use_adv else None
        print(f"G params={count_parameters(self.generator)} loss={args.loss}")
        if self.discriminator is not None:
            print(f"D params={count_parameters(self.discriminator)}")

        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.opt_d = None
        if self.discriminator is not None:
            self.opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.run_dir = os.path.join(args.save_dir, run_name(args.model, args.loss))
        os.makedirs(self.run_dir, exist_ok=True)
        self.best_path = os.path.join(self.run_dir, "best.pth")
        self.best_psnr = -1.0
        self.history = []

    def _masked_inputs(self, image, jitter):
        mask, weight = center_mask(image, self.args.hole, jitter, self.args.overlap)
        masked = fill_hole(image, mask)
        return masked, mask, weight

    def _recon(self, pred, image, weight):
        return ((pred - image) ** 2 * weight).sum() / weight.sum().clamp(min=1)

    def train_model(self):
        for epoch in range(self.args.epochs):
            adv_on = self.discriminator is not None and epoch >= self.warmup_epochs
            self.generator.train()
            if self.discriminator is not None:
                self.discriminator.train()
            recs, advs, dloss = [], [], []
            start = time.time()
            for step, (image, _name) in enumerate(self.train_loader):
                image = image.to(self.device, non_blocking=True)
                masked, mask, weight = self._masked_inputs(image, self.args.jitter)
                pred = self.generator(masked)
                completed = composite(image, pred, mask)
                loss_d = None
                if adv_on:
                    loss_d = self._step_discriminator(image, completed.detach())
                    dloss.append(loss_d)
                loss_g, rec, adv = self._step_generator(pred, image, weight, completed, adv_on)
                recs.append(rec)
                advs.append(adv)
                if step % 20 == 0:
                    extra = "" if loss_d is None else f" d {loss_d:.4f}"
                    print(f"epoch {epoch} step {step} rec {rec:.4f} adv {adv:.4f}{extra}")
            val_stats = self.evaluate(self.val_loader, split="val")
            record = {
                "epoch": epoch,
                "train_rec": float(np.mean(recs) if recs else 0),
                "train_adv": float(np.mean(advs) if advs else 0),
                "train_d": float(np.mean(dloss) if dloss else 0),
                "val_psnr": val_stats["psnr"],
                "val_ssim": val_stats["ssim"],
                "val_mae": val_stats["mae"],
                "val_mse": val_stats["mse"],
            }
            self.history.append(record)
            self._write_history()
            message = (
                f"epoch {epoch} rec {record['train_rec']:.4f} adv {record['train_adv']:.4f} "
                f"val_PSNR {val_stats['psnr']:.2f} val_SSIM {val_stats['ssim']:.4f} "
                f"time {time.time() - start:.1f}s"
            )
            if epoch >= self.warmup_epochs and val_stats["psnr"] > self.best_psnr:
                self.best_psnr = val_stats["psnr"]
                self._save(self.best_path, epoch, val_stats["psnr"])
                message += "  *best"
            if self.warmup_epochs and epoch + 1 == self.warmup_epochs:
                self._save(os.path.join(self.run_dir, "warmup.pth"), epoch, val_stats["psnr"])
                message += "  *warmup"
            print(message)
        print(f"best val PSNR {self.best_psnr:.2f} -> {self.best_path}")
        return self.history

    def _step_discriminator(self, real, fake):
        self.opt_d.zero_grad(set_to_none=True)
        real_logit = self.discriminator(real)
        fake_logit = self.discriminator(fake)
        loss = 0.5 * (
            self.bce(real_logit, torch.full_like(real_logit, 0.9))
            + self.bce(fake_logit, torch.zeros_like(fake_logit))
        )
        loss.backward()
        self.opt_d.step()
        return float(loss.detach())

    def _step_generator(self, pred, image, weight, completed, adv_on):
        self.opt_g.zero_grad(set_to_none=True)
        rec = self._recon(pred, image, weight) if self.use_rec else pred.new_zeros(())
        adv = pred.new_zeros(())
        if adv_on:
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logit = self.discriminator(completed)
            adv = self.bce(logit, torch.ones_like(logit))
            for param in self.discriminator.parameters():
                param.requires_grad = True
        loss = (rec if self.use_rec else 0) + (self.args.lambda_adv * adv if adv_on else 0)
        if adv_on and not self.use_rec:
            loss = adv
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.opt_g.step()
        rec_value = float(rec.detach()) if torch.is_tensor(rec) else 0.0
        adv_value = float(adv.detach()) if torch.is_tensor(adv) else 0.0
        return float(loss.detach()), rec_value, adv_value

    @torch.no_grad()
    def evaluate(self, loader, split="val"):
        self.generator.eval()
        meter = RestorationMeter()
        for image, _name in loader:
            image = image.to(self.device, non_blocking=True)
            masked, mask, _weight = self._masked_inputs(image, jitter=0)
            pred = self.generator(masked)
            meter.update(pred.cpu().numpy(), image.cpu().numpy(), mask.cpu().numpy())
        stats = meter.compute()
        if split != "val":
            print(f"===== {split} =====")
            print(format_metrics(stats))
            path = os.path.join(self.run_dir, f"{split}.metrics.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"split": split, **stats}, f, indent=2)
        self.generator.train()
        return stats

    def _save(self, path, epoch, psnr):
        torch.save({
            "generator": self.generator.state_dict(),
            "epoch": epoch,
            "psnr": psnr,
            "model": self.args.model,
            "loss": self.args.loss,
            "img_size": self.args.img_size,
            "hole": self.args.hole,
            "base_unet": self.args.base_unet,
            "base_context": self.args.base_context,
        }, path)

    def _write_history(self):
        with open(os.path.join(self.run_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
