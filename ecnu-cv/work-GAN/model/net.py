import torch
from torch import nn

try:
    from .block import ResidualConvBlock
except ImportError:
    from block import ResidualConvBlock


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class UNet(nn.Module):
    """课程里的残差 U-Net。跳跃连接把编码器特征拼进解码器。"""

    def __init__(self, in_channels=1, out_channels=1, base=32):
        super(UNet, self).__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16
        self.enc1 = ResidualConvBlock(in_channels, c1)
        self.enc2 = ResidualConvBlock(c1, c2)
        self.enc3 = ResidualConvBlock(c2, c3)
        self.enc4 = ResidualConvBlock(c3, c4)
        self.bottleneck = ResidualConvBlock(c4, c5)
        self.pool = nn.MaxPool2d(2)
        self.upconv4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = ResidualConvBlock(c4 + c4, c4)
        self.upconv3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ResidualConvBlock(c3 + c3, c3)
        self.upconv2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ResidualConvBlock(c2 + c2, c2)
        self.upconv1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ResidualConvBlock(c1 + c1, c1)
        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))
        dec4 = self.dec4(torch.cat((self.upconv4(bottleneck), enc4), dim=1))
        dec3 = self.dec3(torch.cat((self.upconv3(dec4), enc3), dim=1))
        dec2 = self.dec2(torch.cat((self.upconv2(dec3), enc2), dim=1))
        dec1 = self.dec1(torch.cat((self.upconv1(dec2), enc1), dim=1))
        return torch.sigmoid(self.out_conv(dec1))


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


def build_generator(name="context", base_unet=32, base_context=64):
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
