import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

# CamVid 32 类合并为 SegNet 常用的 11 类，Void 不参与损失和指标。
# 名称按字母数字归一化后匹配 codes.txt。
CLASS_NAMES = [
    "Sky",
    "Building",
    "Pole",
    "Road",
    "Pavement",
    "Tree",
    "SignSymbol",
    "Fence",
    "Car",
    "Pedestrian",
    "Bicyclist",
]
CLASS_NAMES_ZH = [
    "天空",
    "建筑",
    "杆柱",
    "道路",
    "人行道",
    "树木",
    "标志",
    "围栏",
    "车辆",
    "行人",
    "骑行者",
]
CLASS_COLORS = np.array(
    [
        [128, 128, 128],
        [128, 0, 0],
        [192, 192, 128],
        [128, 64, 128],
        [0, 0, 192],
        [128, 128, 0],
        [192, 128, 128],
        [64, 64, 128],
        [64, 0, 128],
        [64, 64, 0],
        [0, 128, 192],
    ],
    dtype=np.uint8,
)

_GROUP = {
    "sky": 0,
    "building": 1,
    "wall": 1,
    "archway": 1,
    "bridge": 1,
    "tunnel": 1,
    "columnpole": 2,
    "trafficcone": 2,
    "road": 3,
    "lanemkgsdriv": 3,
    "lanemkgsnondriv": 3,
    "sidewalk": 4,
    "parkingblock": 4,
    "roadshoulder": 4,
    "tree": 5,
    "vegetationmisc": 5,
    "signsymbol": 6,
    "misctext": 6,
    "trafficlight": 6,
    "fence": 7,
    "car": 8,
    "suvpickuptruck": 8,
    "truckbus": 8,
    "train": 8,
    "othermoving": 8,
    "pedestrian": 9,
    "child": 9,
    "cartluggagepram": 9,
    "animal": 9,
    "bicyclist": 10,
    "motorcyclescooter": 10,
    "void": 255,
    "unlabelled": 255,
}


def _norm_name(name):
    return "".join(ch for ch in name.lower() if ch.isalnum())


def colorize(mask):
    mask = np.asarray(mask)
    out = np.zeros(mask.shape[:2] + (3,), dtype=np.uint8)
    for idx, color in enumerate(CLASS_COLORS):
        out[mask == idx] = color
    return out


def _find_dir(root, names):
    for name in names:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    return None


def locate_camvid(root):
    root = os.path.abspath(root)
    candidates = [root, os.path.join(root, "camvid"), os.path.dirname(root)]
    for base in candidates:
        images = _find_dir(base, ["images", "train", "img"])
        labels = _find_dir(base, ["labels", "trainannot", "gt", "masks"])
        # fastai 把 train/val 图像放在同一个 images 目录
        if images and labels and os.path.basename(images) == "images":
            return base, images, labels
    # 已经是 camvid 根目录但上面没命中时，再宽松找一次
    for base in candidates:
        images = _find_dir(base, ["images"])
        labels = _find_dir(base, ["labels"])
        if images and labels:
            return base, images, labels
    raise FileNotFoundError(
        f"在 {root} 下没有找到 CamVid 的 images/ 和 labels/。请先运行 download_camvid.py"
    )


def _read_list(path):
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token = line.split()[0]
            names.append(os.path.basename(token))
    return names


def _label_path(labels_dir, image_name):
    stem, ext = os.path.splitext(image_name)
    candidates = [
        os.path.join(labels_dir, image_name),
        os.path.join(labels_dir, stem + "_P" + ext),
        os.path.join(labels_dir, stem + "_P.png"),
        os.path.join(labels_dir, stem + "_L.png"),
        os.path.join(labels_dir, stem + ".png"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"找不到 {image_name} 对应的标注，已尝试 {candidates}")


def load_codes(root):
    path = os.path.join(root, "codes.txt")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_lut(codes):
    """把原始类别下标映射到 11 类。没有 codes 时认为标注已经是 0–10，255 为忽略。"""
    lut = np.full(256, 255, dtype=np.uint8)
    if codes is None:
        for idx in range(len(CLASS_NAMES)):
            lut[idx] = idx
        return lut
    for idx, name in enumerate(codes):
        key = _norm_name(name)
        if key not in _GROUP:
            raise KeyError(f"codes.txt 中的类别 {name} 没有合并规则")
        lut[idx] = _GROUP[key]
    return lut


def mask_to_index(path):
    """读标注。调色板图或单通道图的像素值就是类别下标；RGB 且三通道相等时同样视为下标。"""
    label = Image.open(path)
    if label.mode in ("P", "L", "I"):
        return np.array(label, dtype=np.uint8)
    arr = np.array(label)
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    if arr.ndim == 3 and np.all(arr[:, :, 0] == arr[:, :, 1]) and np.all(arr[:, :, 1] == arr[:, :, 2]):
        return arr[:, :, 0].astype(np.uint8)
    raise ValueError(f"{path} 是彩色标注，当前加载器只接受类别下标图")


def collect_pairs(root):
    base, images_dir, labels_dir = locate_camvid(root)
    names = sorted(
        name for name in os.listdir(images_dir)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    pairs = [(os.path.join(images_dir, name), _label_path(labels_dir, name), name) for name in names]
    return base, pairs


def make_splits(root, seed=42, val_ratio=0.15):
    """valid.txt 作为最终测试集；其余图像按 seed 划出验证集，剩下的用于训练。"""
    base, pairs = collect_pairs(root)
    by_name = {name: (img, lab) for img, lab, name in pairs}
    valid_file = os.path.join(base, "valid.txt")
    if os.path.isfile(valid_file):
        test_names = _read_list(valid_file)
        missing = [name for name in test_names if name not in by_name]
        if missing:
            raise FileNotFoundError(f"valid.txt 中有 {len(missing)} 个文件不在 images/ 里，例如 {missing[:3]}")
        test_pairs = [by_name[name] for name in test_names]
        rest = [by_name[name] for name in by_name if name not in set(test_names)]
    else:
        names = list(by_name)
        rng = random.Random(seed)
        rng.shuffle(names)
        n_test = max(1, int(round(len(names) * 0.2)))
        test_pairs = [by_name[name] for name in names[:n_test]]
        rest = [by_name[name] for name in names[n_test:]]

    rng = random.Random(seed)
    order = list(range(len(rest)))
    rng.shuffle(order)
    n_val = max(1, int(round(len(rest) * val_ratio)))
    val_pairs = [rest[i] for i in order[:n_val]]
    train_pairs = [rest[i] for i in order[n_val:]]
    return base, train_pairs, val_pairs, test_pairs


class MyDataset(Dataset):
    def __init__(self, pairs, lut, img_size, augment=False):
        self.pairs = list(pairs)
        self.lut = np.asarray(lut, dtype=np.uint8)
        self.img_size = img_size  # (H, W)
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        img_path, mask_path = self.pairs[index]
        image = Image.open(img_path).convert("RGB")
        index_mask = mask_to_index(mask_path)
        mapped = self.lut[index_mask]
        mask = Image.fromarray(mapped, mode="L")

        image = TF.resize(image, self.img_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.img_size, interpolation=TF.InterpolationMode.NEAREST)

        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            angle = random.uniform(-12, 12)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            # 旋转出的空白不能标成类别 0，填 ignore
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=255)
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))

        img_tensor = TF.to_tensor(image)
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))
        return img_tensor, mask_tensor


def compute_class_weights(pairs, lut, num_classes, ignore_index=255):
    """SegNet 的 median frequency balancing：weight_c = median(freq) / freq_c。

    在原始分辨率上统计，避免缩放插值改变像素计数。
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, mask_path in pairs:
        mapped = lut[mask_to_index(mask_path)]
        labels = mapped[mapped != ignore_index]
        if labels.size == 0:
            continue
        counts += np.bincount(labels.astype(np.int64), minlength=num_classes)[:num_classes]
    total = counts.sum()
    freq = counts / max(total, 1.0)
    present = freq > 0
    median = np.median(freq[present]) if present.any() else 1.0
    weights = np.ones(num_classes, dtype=np.float32)
    weights[present] = (median / freq[present]).astype(np.float32)
    return torch.from_numpy(weights), torch.from_numpy(counts)


def build_dataloaders(args, with_weights=True):
    base, train_pairs, val_pairs, test_pairs = make_splits(args.data_root, seed=args.seed)
    lut = build_lut(load_codes(base))
    size = (args.img_height, args.img_width)
    train_set = MyDataset(train_pairs, lut, size, augment=True)
    val_set = MyDataset(val_pairs, lut, size, augment=False)
    test_set = MyDataset(test_pairs, lut, size, augment=False)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    def loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            generator=generator if shuffle else None,
        )

    print(
        f"split train/val/test = {len(train_set)}/{len(val_set)}/{len(test_set)}  "
        f"size={size[0]}x{size[1]}"
    )
    if not with_weights:
        return loader(train_set, True), loader(val_set, False), loader(test_set, False), None, None
    weights, counts = compute_class_weights(train_pairs, lut, args.num_classes, args.ignore_index)
    return loader(train_set, True), loader(val_set, False), loader(test_set, False), weights, counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/camvid")
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--img_width", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--ignore_index", type=int, default=255)
    args = parser.parse_args()
    train_loader, _, _, weights, counts = build_dataloaders(args)
    img, mask = next(iter(train_loader))
    print("batch", tuple(img.shape), tuple(mask.shape), "mask unique", torch.unique(mask).tolist())
    print("class weights", weights.tolist())
    print("pixels", counts.tolist())
