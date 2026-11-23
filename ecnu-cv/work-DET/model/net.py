# -*- coding: utf-8 -*-
"""RetinaNet 模型定义。

改写自课程样例的 model/net.py（原 U-Net 分割网络）。
采用 torchvision 官方 RetinaNet（ResNet50 + FPN + Focal Loss）结构，加载 COCO 预训练权重。
由于 VOC 的 20 个类别均可在 COCO 的 80 类中找到对应项，本实现保持官方 91 类输出不变，
通过数据集与评测模块中的类别映射，把 VOC 标签映射到 COCO 标签空间。
这样既保留了完整的预训练权重，又实现了在 VOC 上的检测。
"""
import torch
from torchvision.models.detection import (
    retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights,
)
from dataset import VOC_CLASSES

COCO_CATEGORIES = RetinaNet_ResNet50_FPN_Weights.COCO_V1.meta["categories"]

# VOC 类别名 -> COCO 类别名（命名不同的类别需要显式对应）
VOC_ALIAS = {
    "aeroplane": "airplane", "motorbike": "motorcycle", "diningtable": "dining table",
    "pottedplant": "potted plant", "sofa": "couch", "tvmonitor": "tv",
}

# VOC 类别下标(0-based) -> COCO 标签(1-based, 0 为背景)
VOC_TO_COCO = {}
COCO_TO_VOC = {}
_voc2idx = {name: i for i, name in enumerate(VOC_CLASSES)}
_coco2idx = {name: idx for idx, name in enumerate(COCO_CATEGORIES)}
for _vname, _vidx0 in _voc2idx.items():
    _cname = VOC_ALIAS.get(_vname, _vname)
    if _cname in _coco2idx:
        _coco_label = _coco2idx[_cname]        # categories 首项为 __background__，索引即 label
        VOC_TO_COCO[_vidx0 + 1] = _coco_label
        COCO_TO_VOC[_coco_label] = _vidx0 + 1


def build_model(pretrained=True, score_thresh=0.05):
    weights = RetinaNet_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
    model = retinanet_resnet50_fpn(weights=weights, num_classes=91,
                                   min_size=600, max_size=800,
                                   box_score_thresh=score_thresh)
    return model


if __name__ == "__main__":
    m = build_model(pretrained=False)
    print("params %.2f M" % (sum(p.numel() for p in m.parameters()) / 1e6))
    print("VOC->COCO mapped classes:", len(VOC_TO_COCO))
