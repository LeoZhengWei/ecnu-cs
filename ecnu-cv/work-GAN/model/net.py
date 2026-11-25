import torch
from torch import nn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ChannelwiseFC(nn.Module):
    """每个特征图内部做全连接，特征图之间不连参数。参数量是 m·n^4。"""

    def __init__(self, channels, size):
        super(ChannelwiseFC, self).__init__()
        self.channels = channels
        self.size = size
        n = size * size
        self.weight = nn.Parameter(torch.empty(channels, n, n))
        self.bias = nn.Parameter(torch.zeros(channels, n))
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / (n ** 0.5))

    def forward(self, x):
        batch, channels, height, width = x.shape
        flat = x.reshape(batch, channels, height * width)
        out = torch.einsum("bcn,cnm->bcm", flat, self.weight) + self.bias
        return out.reshape(batch, channels, height, width)


class ContextEncoder(nn.Module):
    """论文中的编码器-解码器：没有跳跃连接，瓶颈用通道维全连接把整图信息送进解码器。"""

    def __init__(self, in_channels=3, out_channels=3, base=64):
        super(ContextEncoder, self).__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c3, c4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c4, c4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.channel_fc = ChannelwiseFC(c4, size=8)
        self.drop = nn.Dropout(0.2)
        self.mix = nn.Sequential(
            nn.Conv2d(c4, c4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        def up(cin, cout):
            return [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]

        self.decoder = nn.Sequential(
            *up(c4, c4),
            *up(c4, c3),
            *up(c3, c2),
            *up(c2, c1),
            *up(c1, c1),
            nn.Conv2d(c1, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.mix(self.drop(self.channel_fc(feat)))
        return self.decoder(feat)


class Discriminator(nn.Module):
    """看整张补全后的图，不看遮挡位置。输出每个位置的真伪 logit。"""

    def __init__(self, in_channels=3, base=64):
        super(Discriminator, self).__init__()

        def block(cin, cout, normalize):
            layers = [nn.Conv2d(cin, cout, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_channels, base, False),
            *block(base, base * 2, True),
            *block(base * 2, base * 4, True),
            *block(base * 4, base * 8, True),
            nn.Conv2d(base * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def build_generator(name="context", base_context=64):
    name = name.lower()
    if name in ("context", "contextencoder", "ce", "gan"):
        return ContextEncoder(3, 3, base=base_context)
    raise ValueError(name)


if __name__ == "__main__":
    x = torch.randn(2, 3, 256, 256)
    model = build_generator("context")
    y = model(x)
    print("context", count_parameters(model), tuple(y.shape))
    d = Discriminator()
    print("D", count_parameters(d), tuple(d(x).shape))
