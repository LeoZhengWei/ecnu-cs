import torch
import torchvision
from torch.utils.data import Dataset, random_split, DataLoader
import os
from PIL import Image
from torchvision import transforms
# shape = (512, 512)
class MyDataset(Dataset):
    def __init__(self, pairs, lut, img_size, augment=False):
        self.pairs = list(pairs)
        self.lut = np.asarray(lut, dtype=np.uint8)
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        img_path, mask_path = self.pairs[index]
        image = Image.open(img_path).convert('RGB')
        mapped = self.lut[mask_to_index(mask_path)]
        mask = Image.formarray(mapped, mode='L')
        image = TF.resize(image, self.img_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.img_size, interpolation=TF.InterpolationMode.NEAREST)
        if self.augment:
            if random.random() < 0.5:
                image = TF.hfilp(image)
                mask = TF.hfilp(mask)
            angle = random.uniform(-12, 12)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=255)
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
        img_tensor = TF.to_tensor(image)
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))
        return img_tensor, mask_tensor


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default="data/camvid")
    parser.add_argument('--img_height', type=int, default=240)
    parser.add_argument('--img_width', type=int, default=320)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_classes', type=int, default=11)
    parser.add_argument('--ignore_index', type=int, default=255)
    args = parser.parse_args()
    train_loader, _, _, weights, counts = build_dataloaders(args)
    img, mask = next(iter(train_loader))
    print(tuple(img.shape), tuple(mask.shape), torch.unique(mask).tolist())
    print(weights.tolist())
