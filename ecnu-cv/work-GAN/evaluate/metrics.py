"""图像复原指标。像素误差在遮挡区域上计算，结构相似度在贴回原图后的整图上计算。"""

import math

import numpy as np
from scipy.ndimage import gaussian_filter


def mae(pred, target, mask):
    return _masked_mean(np.abs(pred - target), mask)


def mse(pred, target, mask):
    return _masked_mean((pred - target) ** 2, mask)


def psnr(pred, target, mask, peak=1.0):
    error = mse(pred, target, mask)
    if error <= 1e-12:
        return 99.0
    return 10.0 * math.log10((peak * peak) / error)


def _masked_mean(values, mask):
    weight = mask.astype(np.float64)
    denom = float(weight.sum())
    if denom <= 0:
        return float("nan")
    return float((values.astype(np.float64) * weight).sum() / denom)


def _luma(image):
    """RGB 转亮度后再算结构相似度，和人眼看整张图更接近。"""
    if image.ndim == 2:
        return image
    return 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]


def ssim(pred, target, data_range=1.0):
    """整图平均 SSIM。窗口是 11×11 高斯，稳定常数按论文常用的 0.01 和 0.03。"""
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    sigma = 1.5
    mu_x = gaussian_filter(pred, sigma)
    mu_y = gaussian_filter(target, sigma)
    sigma_x = gaussian_filter(pred * pred, sigma) - mu_x * mu_x
    sigma_y = gaussian_filter(target * target, sigma) - mu_y * mu_y
    sigma_xy = gaussian_filter(pred * target, sigma) - mu_x * mu_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return float((num / den).mean())


class RestorationMeter:
    """MAE、MSE、PSNR 在全部测试图的遮挡像素上汇总后再计算，SSIM 按图像平均。"""

    def __init__(self):
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.hole_count = 0.0
        self.full_sq = 0.0
        self.full_count = 0.0
        self.ssim_sum = 0.0
        self.n = 0

    def update(self, pred, target, mask):
        pred = np.asarray(pred, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        mask = np.asarray(mask, dtype=np.float64)
        if pred.ndim == 3:
            pred = pred[None]
            target = target[None]
            mask = mask[None]
        for i in range(pred.shape[0]):
            hole = mask[i, 0] > 0.5 if mask.ndim == 4 else mask[i] > 0.5
            diff = pred[i] - target[i]
            self.abs_sum += np.abs(diff[:, hole]).sum()
            self.sq_sum += np.square(diff[:, hole]).sum()
            self.hole_count += float(hole.sum() * diff.shape[0])
            completed = target[i].copy()
            completed[:, hole] = pred[i][:, hole]
            full = completed - target[i]
            self.full_sq += np.square(full).sum()
            self.full_count += float(full.size)
            self.ssim_sum += ssim(_luma(completed), _luma(target[i]))
            self.n += 1

    def compute(self):
        mse_v = self.sq_sum / max(self.hole_count, 1.0)
        full_mse = self.full_sq / max(self.full_count, 1.0)
        return {
            "mae": self.abs_sum / max(self.hole_count, 1.0),
            "mse": mse_v,
            "psnr": 99.0 if mse_v <= 1e-12 else 10.0 * math.log10(1.0 / mse_v),
            "ssim": self.ssim_sum / max(self.n, 1),
            "psnr_full": 99.0 if full_mse <= 1e-12 else 10.0 * math.log10(1.0 / full_mse),
            "count": self.n,
        }


def format_metrics(stats):
    return (
        f"MAE={stats['mae']:.4f} MSE={stats['mse']:.5f} "
        f"PSNR={stats['psnr']:.2f}dB SSIM={stats['ssim']:.4f} "
        f"fullPSNR={stats['psnr_full']:.2f}dB n={stats['count']}"
    )
