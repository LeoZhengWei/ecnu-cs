"""建筑立面照片。复原的真值就是照片本身，遮挡在训练时按中心区域生成。"""

import os
import random

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


def collect_images(root):
    names = []
    for name in sorted(os.listdir(root)):
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            names.append(os.path.join(root, name))
    return names


def make_splits(root, seed, n_test, n_val):
    files = collect_images(root)
    order = list(files)
    random.Random(seed).shuffle(order)
    test = order[:n_test]
    val = order[n_test:n_test + n_val]
    train = order[n_test + n_val:]

    def pack(paths):
        return [(path, os.path.basename(path)) for path in paths]

    return pack(train), pack(val), pack(test)


def to_square(image, size):
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return TF.resize(image, (size, size), interpolation=TF.InterpolationMode.BILINEAR)


class FacadeDataset(Dataset):
    def __init__(self, pairs, size=256, augment=False):
        self.pairs = list(pairs)
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        path, name = self.pairs[index]
        image = Image.open(path).convert("RGB")
        image = to_square(image, self.size)
        if self.augment and random.random() < 0.5:
            image = TF.hflip(image)
        tensor = TF.to_tensor(image)
        return tensor, name


def build_dataloaders(args):
    train_pairs, val_pairs, test_pairs = make_splits(
        args.data_root, args.seed, args.n_test, args.n_val
    )
    train_set = FacadeDataset(train_pairs, args.img_size, augment=True)
    val_set = FacadeDataset(val_pairs, args.img_size, augment=False)
    test_set = FacadeDataset(test_pairs, args.img_size, augment=False)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    def loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True,
            generator=generator if shuffle else None,
        )

    print(
        f"split train/val/test = {len(train_set)}/{len(val_set)}/{len(test_set)}  "
        f"size={args.img_size}"
    )
    return loader(train_set, True), loader(val_set, False), loader(test_set, False), test_pairs
