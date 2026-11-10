import json
import os

import torch

from dataset import build_dataloaders
from model.net import count_parameters
from model.training import Trainer, run_name, set_seed
import configs.config_loader as cfg_loader


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


if __name__ == "__main__":
    args = cfg_loader.get_config()
    print(f"args={args}")
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    train_loader, val_loader, test_loader, _test_pairs = build_dataloaders(args)
    trainer = Trainer(train_loader, val_loader, args)
    run_dir = os.path.join(args.save_dir, run_name(args.model, args.loss))
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "g_params": count_parameters(trainer.generator)}, f, indent=2)
    trainer.train_model()
    ckpt = torch_load(trainer.best_path)
    trainer.generator.load_state_dict(ckpt["generator"])
    trainer.generator.to(trainer.device)
    trainer.evaluate(test_loader, split="test")
