import configargparse


def config_parser():
    parser = configargparse.ArgumentParser(description="建筑立面中心区域复原")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--warmup_epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_d", type=float, default=2e-5)
    parser.add_argument("--lambda_adv", type=float, default=0.005)
    parser.add_argument("--model", type=str, default="context")
    parser.add_argument("--loss", type=str, default="joint", choices=["l2", "joint", "adv"])
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--hole", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=7)
    parser.add_argument("--jitter", type=int, default=16)
    parser.add_argument("--base_unet", type=int, default=32)
    parser.add_argument("--base_context", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default="data/facades")
    parser.add_argument("--save_dir", type=str, default="runs")
    parser.add_argument("--n_test", type=int, default=40)
    parser.add_argument("--n_val", type=int, default=30)
    return parser


def get_config():
    return config_parser().parse_args()
