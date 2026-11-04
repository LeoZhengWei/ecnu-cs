"""
    语义分割评价指标
    混淆矩阵M的行是真实类别、列是预测类别，M[i, j] 表示真实为i, 预测为j的像素数
    Void / ignore像素不进入矩阵。记
        TP_i = M[i, i]
        GT_i = sum_j M[i, j]
        Pred_i = sum_j M[j, i]
        Union_i = GT_i + Pred_i - TP_i
    则
        IoU_i = TP_i / Union_i
        Precision_i = TP_i / Pred_i
        Recall_i = TP_i / GT_i
        Dice_i = 2 TP_i / (GT_i + Pred_i)
        PA = sum_i TP_i / sum_{i, j} M[i, j]
        mPA = mean_i Recall_i
        mIoU = mean_i IoU_i
        mDice = mean_i Dice_i
        FWIoU = sum_i (GT_i / N) * IoU_i
某个类别在该划分中没有像素时，对应为NaN，均值使用nanmean，不把空类别算成0
"""

import numpy as np

class ConfusionMeter:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.cm = np.zeros((num_classes, num_classes))

    def reset(self):
        self.cm[:] = 0

    def update(self, pred, target):
        pred = np.asarray(pred).reshape(-1)
        target = np.asarray(target).reshape(-1)
        valid = (target != self.ignore_index)
        pred = pred[valid].astype(np.int64)
        target = target[valid].astype(np.int64)
        in_range = (pred >= 0) & (pred < self.num_classes) & (target >= 0) & (target < self.num_classes)
        pred = pred[in_range]
        target = target[in_range]
        if pred.size == 0:
            return
        k = self.num_classes
        hist = np.bincount(k * target + pred, minlength=k*k)
        self.cm += hist.reshape(k, k)

    def compute(self):
        cm = self.cm.astype(np.float64)
        tp = np.diag(cm)
        gt = cm.sum(axis=1)
        pred_sum = cm.sum(axis=0)
        union = gt + pred_sum - tp
        total = cm.sum()

        def safe_div(num, den):
            out = np.full_like(num, np.nan, dtype=np.float64)
            mask = den > 0
            out[mask] = num[mask] / den[mask]
            return out

        iou = safe_div(tp, union)
        precision = safe_div(tp, pred_sum)
        recall = safe_div(tp, gt)
        dice = safe_div(2.0 * tp, gt + pred_sum)
        pixel_acc = float(tp.sum() / total) if total > 0 else 0.0
        freq = gt /total if total > 0 else np.zeros_like(gt)
        present = ~np.isnan(iou)
        fwiou = float(np.sum(freq[present] * iou[present])) if present.any() else 0.0

        return {
            "pixel_acc": pixel_acc,
            "mean_acc": float(np.nanmean(recall)) if np.any(~np.isnan(recall)) else 0.0,
            "miou": float(np.nanmean(iou)) if present.any() else 0.0,
            "fwiou": fwiou,
            "mdice": float(np.nanmean(dice)) if np.any(~np.isnan(dice)) else 0.0, 
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "dice": dice,
            "support": gt.astype(np.int64),
            "confusion": self.cm.copy(),
        }

def format_metrics(stats, class_names):
    lines = []
    header = f"{'class': <16}{'IoU':>8}{'Prec':>8}{'Recall':>8}{'pixles':>12}"
    lines.append(header)
    for i, name in enumerate(class_names):
        def cell(arr):
            value = arr[i]
            return " nan" if np.isnan(value) else f"{value:8.4f}"

        lines.append(
            f"{name: <16}{cell(stats['iou'])}{cell(stats['precision'])}"
            f"{cell(stats['recall'])}{cell(stats['dice'])}{stats['support'][i]:12d}"
        )
    lines.append(
        "PA={pixel_acc:.4f} mPA={mean_acc:.4f} mIoU={miou:.4f} "
        "FWIoU={fwiou:.4f} mDice={mdice:.4f}".format(**stats)
    )
    return "\n".join(lines)

if __name__ == "__main__":
    meter = ConfusionMeter(num_classes=11, ignore_index=255)
    target = np.array([[0, 0, 1, 255], [1, 2, 2, 2]])
    pred = np.array([[0, 1, 1, 0], [1, 2, 0, 2]])
    meter.update(pred, target)
    stats = meter.compute()
    assert abs(stats["pixel_acc"] - 5 / 7) < 1e-8, stats["pixel_acc"]
    assert abs(stats["iou"][0] - 1 / 3) < 1e-8
    assert abs(stats["iou"][1] - 2 / 3) < 1e-8
    assert abs(stats["iou"][2] - 2 / 3) < 1e-8
    print(format_metrics(stats, ["a", "b", "c"]))
    print("metrics self-check ok")