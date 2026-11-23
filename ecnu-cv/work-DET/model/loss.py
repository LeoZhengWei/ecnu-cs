# -*- coding: utf-8 -*-
"""RetinaNet 的损失：Focal Loss（分类）+ Smooth L1（框回归），以及锚框匹配。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

BBOX_REG_WEIGHTS = (0.1, 0.1, 0.2, 0.2)
POS_IOU = 0.5
NEG_IOU = 0.4


def box_iou(a, b):
    """两组框的 IoU，a:[N,4] b:[M,4]（角点式），返回 [N,M]。"""
    inter_l = torch.max(a[:, None, 0], b[None, :, 0])
    inter_t = torch.max(a[:, None, 1], b[None, :, 1])
    inter_r = torch.min(a[:, None, 2], b[None, :, 2])
    inter_b = torch.min(a[:, None, 3], b[None, :, 3])
    inter = (inter_r - inter_l).clamp(min=0) * (inter_b - inter_t).clamp(min=0)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


def encode_box(anchors_cxcywh, gt_corners):
    """把真值框编码成相对锚框的偏移，并做标准差归一化。"""
    ax, ay, aw, ah = anchors_cxcywh.unbind(1)
    gx = (gt_corners[:, 0] + gt_corners[:, 2]) / 2
    gy = (gt_corners[:, 1] + gt_corners[:, 3]) / 2
    gw = gt_corners[:, 2] - gt_corners[:, 0]
    gh = gt_corners[:, 3] - gt_corners[:, 1]
    dx = (gx - ax) / aw
    dy = (gy - ay) / ah
    dw = torch.log(gw / aw)
    dh = torch.log(gh / ah)
    return torch.stack([dx / BBOX_REG_WEIGHTS[0], dy / BBOX_REG_WEIGHTS[1],
                        dw / BBOX_REG_WEIGHTS[2], dh / BBOX_REG_WEIGHTS[3]], dim=1)


def assign_targets(anchor_corners, anchor_cxcywh, gt_boxes, gt_labels, batch_size,
                   device, max_pos=128):
    """为每张图的锚框分配类别 id（0=背景,>0=前景,-1=忽略）与回归目标。"""
    all_ids, all_reg, all_pos = [], [], []
    n = anchor_corners.shape[0]
    for b in range(batch_size):
        gt = gt_boxes[b]
        labels_gt = gt_labels[b]
        ids = torch.full((n,), -1, dtype=torch.long, device=device)
        reg = torch.zeros((n, 4), dtype=torch.float32, device=device)
        pos_mask = torch.zeros(n, dtype=torch.bool, device=device)
        if gt.numel() == 0:
            ids[:] = 0
            all_ids.append(ids); all_reg.append(reg); all_pos.append(pos_mask)
            continue
        ious = box_iou(anchor_corners, gt)                       # [N, M]
        best_iou, best_gt = ious.max(dim=1)
        ids[best_iou >= POS_IOU] = 1
        ids[best_iou < NEG_IOU] = 0
        matched_best = ious.argmax(dim=0)
        for gi in range(gt.shape[0]):
            ids[matched_best[gi]] = 1
            best_gt[matched_best[gi]] = gi
        pos_idx = torch.where(ids == 1)[0]
        if pos_idx.numel() > max_pos:
            perm = torch.randperm(pos_idx.numel(), device=device)[:max_pos]
            keep = pos_idx[perm]
            mask = torch.ones(n, dtype=torch.bool, device=device)
            mask[pos_idx] = False
            mask[keep] = True
            ids[~mask] = -1
            pos_idx = keep
        neg_idx = torch.where(ids == 0)[0]
        n_neg = min(neg_idx.numel(), pos_idx.numel() * 3)
        if neg_idx.numel() > n_neg and n_neg > 0:
            drop = neg_idx[torch.randperm(neg_idx.numel(), device=device)[:n_neg]]
            ids[drop] = -1
        for ai in pos_idx:
            ids[ai] = labels_gt[best_gt[ai]]
            pos_mask[ai] = True
        if pos_idx.numel() > 0:
            reg[pos_idx] = encode_box(anchor_cxcywh[pos_idx], gt[best_gt[pos_idx]])
        all_ids.append(ids); all_reg.append(reg); all_pos.append(pos_mask)
    return torch.stack(all_ids), torch.stack(all_reg), torch.stack(all_pos)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, cls_logits, targets, pos_mask):
        """cls_logits:[B,N,C] targets:[B,N] 类别id pos_mask:[B,N] 前景标记。
        背景锚框对所有类别目标都是 0；忽略锚框(target=-1)不参与。"""
        b, n, c = cls_logits.shape
        tgt = targets.reshape(-1).clamp(min=0)
        valid = (targets.reshape(-1) >= 0).float()
        # 前景 -> 对应类 one-hot；背景 -> 全 0
        onehot = F.one_hot(tgt, num_classes=c).float() * pos_mask.reshape(-1, 1).float()
        p = torch.sigmoid(cls_logits.reshape(-1, c))
        ce = F.binary_cross_entropy_with_logits(cls_logits.reshape(-1, c), onehot, reduction="none")
        p_t = p * onehot + (1 - p) * (1 - onehot)
        alpha_t = self.alpha * onehot + (1 - self.alpha) * (1 - onehot)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * ce
        n_pos = pos_mask.sum().clamp(min=1).float()
        return (loss.sum(dim=1) * valid).sum() / n_pos


def smooth_l1_loss(pred, target, pos_mask, beta=1.0 / 9):
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    mask = pos_mask.unsqueeze(-1).expand_as(loss).float()
    return (loss * mask).sum() / pos_mask.sum().clamp(min=1)
