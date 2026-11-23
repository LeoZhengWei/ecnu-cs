# -*- coding: utf-8 -*-
"""把网络输出解码成检测框（锚框偏移 -> 角点坐标）+ 非极大值抑制。"""
import torch
from torchvision.ops import nms

BBOX_REG_WEIGHTS = (0.1, 0.1, 0.2, 0.2)


def decode_boxes(anchors_cxcywh, deltas):
    ax, ay, aw, ah = anchors_cxcywh.unbind(1)
    dx = deltas[:, 0] * BBOX_REG_WEIGHTS[0]
    dy = deltas[:, 1] * BBOX_REG_WEIGHTS[1]
    dw = deltas[:, 2] * BBOX_REG_WEIGHTS[2]
    dh = deltas[:, 3] * BBOX_REG_WEIGHTS[3]
    cx = dx * aw + ax
    cy = dy * ah + ay
    w = torch.exp(dw) * aw
    h = torch.exp(dh) * ah
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)


def postprocess(cls_logits, reg_pred, anchors_cxcywh, score_thresh=0.05,
                topk=1000, nms_thresh=0.5, max_det=100):
    """单张图的后处理，返回 boxes[N,4], scores[N], labels[N]（类别 id 从 1 开始）。"""
    scores = torch.sigmoid(cls_logits)
    # 去掉背景类 0
    scores = scores[:, 1:]
    labels = torch.arange(1, scores.shape[1] + 1, device=scores.device).unsqueeze(0).expand_as(scores)
    flat_score = scores.reshape(-1)
    flat_label = labels.reshape(-1)
    keep = torch.topk(flat_score, min(topk, flat_score.numel())).indices
    sel_score = flat_score[keep]
    sel_label = flat_label[keep]
    boxes_all = decode_boxes(anchors_cxcywh, reg_pred)
    num_cls = scores.shape[1]
    sel_boxes = boxes_all[keep // num_cls]
    # 过滤低分
    mask = sel_score >= score_thresh
    sel_boxes, sel_score, sel_label = sel_boxes[mask], sel_score[mask], sel_label[mask]
    if sel_boxes.numel() == 0:
        empty = torch.zeros((0, 4), device=sel_boxes.device)
        return empty, torch.zeros(0, device=sel_boxes.device), torch.zeros(0, dtype=torch.long, device=sel_boxes.device)
    # 逐类 NMS
    final_boxes, final_scores, final_labels = [], [], []
    for c in sel_label.unique():
        idx = torch.where(sel_label == c)[0]
        b = sel_boxes[idx]
        s = sel_score[idx]
        keep_nms = nms(b, s, nms_thresh)
        final_boxes.append(b[keep_nms])
        final_scores.append(s[keep_nms])
        final_labels.append(torch.full((keep_nms.numel(),), c, device=sel_boxes.device))
    boxes = torch.cat(final_boxes)
    scores = torch.cat(final_scores)
    lab = torch.cat(final_labels)
    # 按分数排序，最多保留 max_det
    order = scores.argsort(descending=True)[:max_det]
    return boxes[order], scores[order], lab[order]
