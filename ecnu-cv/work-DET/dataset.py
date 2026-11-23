# -*- coding: utf-8 -*-
"""VOC2007 目标检测数据集。

改写自课程样例的 dataset.py：原样例读「图像 + 掩码」做分割，
这里改为读「图像 + 边界框 + 类别」做检测。由于采用官方 91 类 RetinaNet，
VOC 标签会被映射到 COCO 标签空间（VOC 的 20 类都能在 COCO 中找到对应项）。
"""
import os
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
CLASS_TO_IDX = {name: i + 1 for i, name in enumerate(VOC_CLASSES)}

# VOC 标签(1-20) -> COCO 标签(1-90)。延迟导入避免循环依赖。
_VOC_TO_COCO = None


def get_voc_to_coco():
    global _VOC_TO_COCO
    if _VOC_TO_COCO is None:
        from model.net import VOC_TO_COCO
        _VOC_TO_COCO = VOC_TO_COCO
    return _VOC_TO_COCO


class VOCDetection(Dataset):
    def __init__(self, root, image_set="trainval", year="2007", augment=False):
        self.root = root
        self.year = year
        self.augment = augment
        base = os.path.join(root, "VOC" + year)
        self.img_dir = os.path.join(base, "JPEGImages")
        self.ann_dir = os.path.join(base, "Annotations")
        split_file = os.path.join(base, "ImageSets", "Main", image_set + ".txt")
        with open(split_file, "r", encoding="utf-8") as f:
            self.ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.ids)

    def _parse_ann(self, img_id):
        tree = ET.parse(os.path.join(self.ann_dir, img_id + ".xml"))
        boxes, labels = [], []
        m = get_voc_to_coco()
        for obj in tree.findall("object"):
            name = obj.find("name").text.strip()
            if name not in CLASS_TO_IDX:
                continue
            bnd = obj.find("bndbox")
            x1 = float(bnd.find("xmin").text); y1 = float(bnd.find("ymin").text)
            x2 = float(bnd.find("xmax").text); y2 = float(bnd.find("ymax").text)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(m.get(CLASS_TO_IDX[name], 0))  # 映射到 COCO 标签
        return boxes, labels

    def __getitem__(self, index):
        img_id = self.ids[index]
        image = Image.open(os.path.join(self.img_dir, img_id + ".jpg")).convert("RGB")
        boxes, labels = self._parse_ann(img_id)

        w, h = image.size
        if self.augment and torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
            boxes = [[w - x2, y1, w - x1, y2] for (x1, y1, x2, y2) in boxes]

        target_h, target_w = 600, 800
        sx, sy = target_w / w, target_h / h
        image = TF.to_tensor(TF.resize(image, (target_h, target_w)))

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            bt = torch.tensor(boxes, dtype=torch.float32)
            bt[:, 0::2] *= sx
            bt[:, 1::2] *= sy
            boxes_t = bt
            labels_t = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([index]),
            "area": (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1]),
            "iscrowd": torch.zeros(len(boxes_t), dtype=torch.int64),
        }
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    ds = VOCDetection("/home/hjadmin/zw/ecnu/ecnu-cv/work-DET/datasets/voc2007/VOCdevkit", "trainval")
    print("samples:", len(ds))
    img, tgt = ds[0]
    print("image:", img.shape, "boxes:", tgt["boxes"].shape, "labels(COCO):", tgt["labels"].tolist())
