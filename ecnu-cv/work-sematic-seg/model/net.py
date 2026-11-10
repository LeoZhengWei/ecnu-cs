import torch
from torch import nn
import torch.nn.functional as F

try:
    from .block import ResidualConvBlock
except ImportError:
    from block import ResidualConvBlock

import os
savepath='features'
if not os.path.exists(savepath):
    os.mkdir(savepath)

def draw_features(width,height,x,savename):
    tic=time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width*height):
        plt.subplot(height,width, i + 1)
        plt.axis('off')
        # plt.tight_layout()
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = (img - pmin) / (pmax - pmin + 0.000001)
        plt.imshow(img, cmap='gray')
        print("{}/{}".format(i,width*height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time()-tic))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base=64):
        super(UNet, self).__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16
        # 上采样
        self.enc1 = self.conv_block(in_channels, c1)
        self.enc2 = self.conv_block(c1, c2)
        self.enc3 = self.conv_block(c2, c3)
        self.enc4 = self.conv_block(c3, c4)
        self.bottleneck = self.conv_block(c4, c5)
        self.upconv4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        # 下采样
        self.dec4 = self.conv_block(c4 + c4, c4)
        self.upconv3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(c3 + c3, c3)
        self.upconv2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(c2 + c2, c2)
        self.upconv1 = nn.ConvTranspose2d(c2, c1, kernel_size=2,stride=2)
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

class SegNet(nn.Module):
    """用最大池化的位置索引做反池化，不拼接编码器特征。"""

    def __init__(self, in_channels, out_channels, base=64):
        super(SegNet, self).__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16
        self.enc1 = ResidualConvBlock(in_channels, c1)
        self.enc2 = ResidualConvBlock(c1, c2)
        self.enc3 = ResidualConvBlock(c2, c3)
        self.enc4 = ResidualConvBlock(c3, c4)
        self.bottleneck = ResidualConvBlock(c4, c5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # 反池化要求通道数和当初被池化的特征一致，先从 1024 压回 512。
        self.reduce = nn.Conv2d(c5, c4, kernel_size=1)
        self.dec4 = ResidualConvBlock(c4, c3)
        self.dec3 = ResidualConvBlock(c3, c2)
        self.dec2 = ResidualConvBlock(c2, c1)
        self.dec1 = ResidualConvBlock(c1, c1)
        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        p1, idx1 = self.pool(enc1)
        enc2 = self.enc2(p1)
        p2, idx2 = self.pool(enc2)
        enc3 = self.enc3(p2)
        p3, idx3 = self.pool(enc3)
        enc4 = self.enc4(p3)
        p4, idx4 = self.pool(enc4)
        bottleneck = self.reduce(self.bottleneck(p4))
        dec4 = self.dec4(self.unpool(bottleneck, idx4, output_size=enc4.size()))
        dec3 = self.dec3(self.unpool(dec4, idx3, output_size=enc3.size()))
        dec2 = self.dec2(self.unpool(dec3, idx2, output_size=enc2.size()))
        dec1 = self.dec1(self.unpool(dec2, idx1, output_size=enc1.size()))
        return self.out_conv(dec1)


class ASPP(nn.Module):
    """空洞空间金字塔。瓶颈只有 15×20，膨胀率用 2/4/6，避免感受野超出特征图。"""

    def __init__(self, in_channels, out_channels, rates=(2, 4, 6)):
        super(ASPP, self).__init__()
        branches = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in rates:
            branches.append(nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3,
                    padding=rate, dilation=rate, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ))
        self.branches = nn.ModuleList(branches)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        n = len(branches) + 1
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * n, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        height, width = x.shape[-2:]
        parts = [branch(x) for branch in self.branches]
        pooled = self.image_pool(x)
        parts.append(F.interpolate(pooled, size=(height, width), mode="bilinear", align_corners=False))
        return self.project(torch.cat(parts, dim=1))


class DeepLabV3Plus(nn.Module):
    """同一套残差编码器，瓶颈接 ASPP，再与 1/4 分辨率特征拼接后上采样。"""

    def __init__(self, in_channels, out_channels, base=64):
        super(DeepLabV3Plus, self).__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16
        self.enc1 = ResidualConvBlock(in_channels, c1)
        self.enc2 = ResidualConvBlock(c1, c2)
        self.enc3 = ResidualConvBlock(c2, c3)
        self.enc4 = ResidualConvBlock(c3, c4)
        self.bottleneck = ResidualConvBlock(c4, c5)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.aspp = ASPP(c5, 256, rates=(2, 4, 6))
        self.low_proj = nn.Sequential(
            nn.Conv2d(c3, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            ResidualConvBlock(256 + 48, 256),
            ResidualConvBlock(256, 128),
        )
        self.out_conv = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))
        high = self.aspp(bottleneck)
        high = F.interpolate(high, size=enc3.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.decoder(torch.cat((high, self.low_proj(enc3)), dim=1))
        fused = F.interpolate(fused, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.out_conv(fused)


class AttentionGate(nn.Module):
    """用解码器特征给跳跃连接打一个空间权重。"""

    def __init__(self, channels):
        super(AttentionGate, self).__init__()
        inter = max(channels // 2, 16)
        self.theta = nn.Conv2d(channels, inter, kernel_size=1, bias=False)
        self.phi = nn.Conv2d(channels, inter, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(inter, 1, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, skip, gate):
        score = self.relu(self.theta(skip) + self.phi(gate))
        weight = torch.sigmoid(self.psi(score))
        return skip * weight


class AttentionUNet(nn.Module):
    """残差 U-Net，拼接前用注意力门控筛选编码器特征。"""

    def __init__(self, in_channels, out_channels, base=64):
        super(AttentionUNet, self).__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16
        self.enc1 = ResidualConvBlock(in_channels, c1)
        self.enc2 = ResidualConvBlock(c1, c2)
        self.enc3 = ResidualConvBlock(c2, c3)
        self.enc4 = ResidualConvBlock(c3, c4)
        self.bottleneck = ResidualConvBlock(c4, c5)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.upconv4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.att4 = AttentionGate(c4)
        self.dec4 = ResidualConvBlock(c4 + c4, c4)
        self.upconv3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.att3 = AttentionGate(c3)
        self.dec3 = ResidualConvBlock(c3 + c3, c3)
        self.upconv2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(c2)
        self.dec2 = ResidualConvBlock(c2 + c2, c2)
        self.upconv1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.att1 = AttentionGate(c1)
        self.dec1 = ResidualConvBlock(c1 + c1, c1)
        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))
        g4 = self.upconv4(bottleneck)
        dec4 = self.dec4(torch.cat((g4, self.att4(enc4, g4)), dim=1))
        g3 = self.upconv3(dec4)
        dec3 = self.dec3(torch.cat((g3, self.att3(enc3, g3)), dim=1))
        g2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat((g2, self.att2(enc2, g2)), dim=1))
        g1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat((g1, self.att1(enc1, g1)), dim=1))
        return self.out_conv(dec1)


def build_model(name, in_channels, out_channels, base=64):
    name = name.lower()
    if name in ("unet", "resunet"):
        return UNet(in_channels, out_channels, base=base)
    if name in ("fcn", "encoderdecoder"):
        return FCN(in_channels, out_channels, base=base)
    if name == "segnet":
        return SegNet(in_channels, out_channels, base=base)
    if name in ("deeplab", "deeplabv3", "deeplabv3plus"):
        return DeepLabV3Plus(in_channels, out_channels, base=base)
    if name in ("attunet", "attention", "attentionunet"):
        return AttentionUNet(in_channels, out_channels, base=base)
    raise ValueError(name)

if __name__ == "__main__":
    x = torch.randn((2, 3, 240, 320))
    for name in ("unet", "fcn", "segnet", "deeplab", "attunet"):
        model = build_model(name, 3, 11)
        y = model(x)
        print(name, count_parameters(model), tuple(y.shape))