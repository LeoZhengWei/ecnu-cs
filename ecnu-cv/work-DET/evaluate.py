# -*- coding: utf-8 -*-
"""PASCAL VOC mAP@IoU=0.5 评测（11-point interpolation）。

数据在 COCO 标签空间处理，避免 VOC<->COCO 映射歧义：
GT 与预测都是 COCO 标签，只对 VOC 涉及的 20 个 COCO 类别计算 AP，再取平均。
"""
import torch
from dataset import VOC_CLASSES
from model.net import VOC_TO_COCO

# VOC 20 类对应的 COCO 类别标签（按 VOC 类别顺序排列，用于报告逐类 AP）
VOC_COCO_IDS = [VOC_TO_COCO[i + 1] for i in range(len(VOC_CLASSES))]


def box_iou(a, b):
    """a:[1,4] 单个检测框；b:[M,4] 多个真值框；返回 [M]。"""
    ix1 = torch.max(a[0, 0], b[:, 0]); iy1 = torch.max(a[0, 1], b[:, 1])
    ix2 = torch.min(a[0, 2], b[:, 2]); iy2 = torch.min(a[0, 3], b[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    aa = (a[0, 2] - a[0, 0]) * (a[0, 3] - a[0, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa + bb - inter).clamp(min=1e-6)


def compute_ap(dets, gt_boxes, n_gt, iou_thresh=0.5):
    if n_gt == 0:
        return 0.0
    order = sorted(dets, key=lambda x: -x[0])
    tp = torch.zeros(len(order)); fp = torch.zeros(len(order)); matched = {}
    for i, (score, img_idx, box) in enumerate(order):
        gbs = gt_boxes.get(img_idx, [])
        if not gbs:
            fp[i] = 1; continue
        ious = box_iou(box.unsqueeze(0), torch.stack(gbs))
        best, bi = ious.max(dim=0)
        if best >= iou_thresh and int(bi) not in matched.get(img_idx, set()):
            tp[i] = 1
            matched.setdefault(img_idx, set()).add(int(bi))
        else:
            fp[i] = 1
    cum_tp = torch.cumsum(tp, 0); cum_fp = torch.cumsum(fp, 0)
    recall = cum_tp / n_gt
    precision = cum_tp / (cum_tp + cum_fp).clamp(min=1e-6)
    return sum((precision[recall >= t].max().item() if (recall >= t).any() else 0.0)
               for t in torch.linspace(0, 1, 11)) / 11


@torch.no_grad()
def compute_map(model, test_loader, device, score_thresh=0.05, iou_thresh=0.5):
    """返回 mAP、按 VOC 类别名索引的每类 AP、每类真值数。"""
    target_ids = set(VOC_COCO_IDS)
    dets = {c: [] for c in target_ids}
    gt_boxes = {c: dict() for c in target_ids}
    n_gt = {c: 0 for c in target_ids}
    model.eval()
    for img_idx, (images, targets) in enumerate(test_loader):
        image = images[0].to(device)
        out = model([image])[0]
        boxes = out["boxes"].cpu(); scores = out["scores"].cpu(); labels = out["labels"].cpu()
        tb = targets[0]["boxes"].cpu(); tl = targets[0]["labels"].cpu()
        for gi in range(tl.shape[0]):
            c = int(tl[gi])
            if c in target_ids:
                gt_boxes[c].setdefault(img_idx, []).append(tb[gi])
                n_gt[c] += 1
        for k in range(scores.shape[0]):
            if float(scores[k]) < score_thresh:
                continue
            c = int(labels[k])
            if c in target_ids:
                dets[c].append((float(scores[k]), img_idx, boxes[k]))
    ap_by_coco = {c: compute_ap(dets[c], gt_boxes[c], n_gt[c], iou_thresh) for c in target_ids}
    ap_per_class = {name: ap_by_coco[cid] for name, cid in zip(VOC_CLASSES, VOC_COCO_IDS)}
    mAP = sum(ap_per_class.values()) / len(VOC_CLASSES)
    return {"mAP": mAP, "ap_per_class": ap_per_class,
            "n_gt": {name: n_gt[cid] for name, cid in zip(VOC_CLASSES, VOC_COCO_IDS)}}
