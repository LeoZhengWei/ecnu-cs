import configargparse


def config_parser():
    parser = configargparse.ArgumentParser(description="CamVid 语义分割")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--model", type=str, default="unet")
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--img_height", type=int, default=240)
    parser.add_argument("--img_width", type=int, default=320)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default="data/camvid")
    parser.add_argument("--save_dir", type=str, default="runs")
    parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--ignore_index", type=int, default=255)
    return parser


def get_config():
    return config_parser().parse_args()