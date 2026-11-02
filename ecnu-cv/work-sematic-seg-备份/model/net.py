import torch
from torch import nn

try:
    from .block import ResidualConvBlock
except ImportError:
    from block import ResidualConvBlock


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class UNet(nn.Module):
    """课程样例中的残差 U-Net。

    编码器四级下采样，瓶颈层，解码器用转置卷积上采样，并把同尺度编码特征
    按通道拼接（跳跃连接）。多类语义分割输出的是 logits，不再接 sigmoid。
    """

    def __init__(self, in_channels, out_channels, base=64):
        super(UNet, self).__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16
        self.enc1 = self.conv_block(in_channels, c1)
        self.enc2 = self.conv_block(c1, c2)
        self.enc3 = self.conv_block(c2, c3)
        self.enc4 = self.conv_block(c3, c4)
        self.bottleneck = self.conv_block(c4, c5)

        self.upconv4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(c4 + c4, c4)
        self.upconv3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(c3 + c3, c3)
        self.upconv2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(c2 + c2, c2)
        self.upconv1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(c1 + c1, c1)
        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def conv_block(self, in_channels, out_channels):
        return ResidualConvBlock(in_channels, out_channels)

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
        return self.out_conv(dec1)


class FCN(nn.Module):
    """对照模型：与 U-Net 相同的残差编码器和转置卷积解码器，但去掉跳跃连接。

    解码器只接收上一层上采样结果，对应 FCN 里“编码后再逐步上采样”的思路。
    这样和 U-Net 的差别集中在有没有跳跃连接，便于比较。
    """

    def __init__(self, in_channels, out_channels, base=64):
        super(FCN, self).__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16
        self.enc1 = ResidualConvBlock(in_channels, c1)
        self.enc2 = ResidualConvBlock(c1, c2)
        self.enc3 = ResidualConvBlock(c2, c3)
        self.enc4 = ResidualConvBlock(c3, c4)
        self.bottleneck = ResidualConvBlock(c4, c5)

        self.upconv4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = ResidualConvBlock(c4, c4)
        self.upconv3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ResidualConvBlock(c3, c3)
        self.upconv2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ResidualConvBlock(c2, c2)
        self.upconv1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ResidualConvBlock(c1, c1)
        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.dec4(self.upconv4(bottleneck))
        dec3 = self.dec3(self.upconv3(dec4))
        dec2 = self.dec2(self.upconv2(dec3))
        dec1 = self.dec1(self.upconv1(dec2))
        return self.out_conv(dec1)


def build_model(name, in_channels, out_channels, base=64):
    name = name.lower()
    if name in ("unet", "resunet"):
        return UNet(in_channels, out_channels, base=base)
    if name in ("fcn", "encoderdecoder"):
        return FCN(in_channels, out_channels, base=base)
    raise ValueError(f"未知模型 {name}，可选 unet 或 fcn")


if __name__ == "__main__":
    x = torch.randn(2, 3, 256, 320)
    for name in ("unet", "fcn"):
        model = build_model(name, in_channels=3, out_channels=11)
        y = model(x)
        print(name, "params", count_parameters(model), "out", tuple(y.shape))
