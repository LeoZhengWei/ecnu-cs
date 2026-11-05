import torch
from torch import nn

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

def build_model(name, in_channels, out_channels, base=64):
    name = name.lower()
    if name in ("unet", "resunet"):
        return UNet(in_channels, out_channels, base=base)
    if name in ("fcn", "encoderdecoder"):
        return FCN(in_channels, out_channels, base=base)
    raise ValueError(name)

if __name__ == "__main__":
    x = torch.randn((2, 3, 240, 320))
    for name in ("unet", "fcn"):
        model = build_model(name, 3, 11)
        y = model(x)
        print(name, count_parameters(model), tuple(y.shape))