import json
import os

import torch

from dataset import CLASS_NAMES, build_dataloaders
from model.net import build_model, count_parameters
from model.training import Trainer, set_seed
import configs.config_loader as cfg_loader


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


if __name__ == "__main__":
    args = cfg_loader.get_config()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, val_loader, test_loader, class_weights, class_counts = build_dataloaders(args)
    model = build_model(args.model, in_channels=3, out_channels=args.num_classes, base=args.base_channels)
    n_params = count_parameters(model)
    print(f"model={args.model}  params={n_params}  class_weights={class_weights.tolist()}")

    run_dir = os.path.join(args.save_dir, args.model.lower())
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "class_weights.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "names": CLASS_NAMES,
                "weights": class_weights.tolist(),
                "train_pixels": [int(v) for v in class_counts.tolist()],
                "params": n_params,
                "config": vars(args),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    trainer = Trainer(train_loader, model, args, val_loader=val_loader, class_weights=class_weights)
    trainer.train_model()

    # 用验证集 mIoU 最好的权重在测试集上算一次，测试集不参与选模型
    ckpt = torch_load(trainer.best_path)
    model.load_state_dict(ckpt["state_dict"])
    trainer.model = model.to(trainer.device)
    trainer.evaluate(test_loader, split="test")
