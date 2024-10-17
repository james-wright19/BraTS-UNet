from torch import nn
import torch
from torchsummary import summary

def crop_tensor(tensor, target_tensor):
    target_size = target_tensor.size()[2:]
    tensor_size = tensor.size()[2:]
    
    diff_h = tensor_size[0] - target_size[0]
    diff_w = tensor_size[1] - target_size[1]
    
    tensor = tensor[:, :, diff_h // 2:(diff_h // 2) + target_size[0], diff_w // 2:(diff_w // 2) + target_size[1]]
    
    return tensor


class ResConvBlock2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, padding=0):
        super(ResConvBlock2D, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=[3,3], padding=padding)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=[3,3], padding=padding)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.adjust_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        skip = x

        x = self.relu(self.bn(self.conv1(x)))
        x = self.relu(self.bn(self.conv2(x)))

        x = x + self.adjust_channels(skip)

        return x

class DownBlock2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, padding=False):
        super(DownBlock2D, self).__init__()

        if padding:
            self.conv = ResConvBlock2D(in_channels, out_channels, 1)
        else: 
            self.conv = ResConvBlock2D(in_channels, out_channels, 0)
        self.max_pool = nn.MaxPool2d([2,2])

    def forward(self, x):
        x = self.conv(x)
        pooled = self.max_pool(x)

        return pooled, x
    
class UpBlock2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, padding=False):
        super(UpBlock2D, self).__init__()

        self.up_conv = nn.ConvTranspose2d(in_channels, in_channels, [2,2], 2)
        if padding:
            self.conv = ResConvBlock2D(in_channels+out_channels, out_channels, 1)
        else:
            self.conv = ResConvBlock2D(in_channels+out_channels, out_channels, 0)
        

    def forward(self, x, skip):
        x = self.up_conv(x)

        skip = crop_tensor(skip, x)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        return x


class ResUNet2D(torch.nn.Module):


    def __init__(self, padding=False):
        super(ResUNet2D, self).__init__()

        self.dec1 = DownBlock2D(1, 64, padding)
        self.dec2 = DownBlock2D(64, 128, padding)
        self.dec3 = DownBlock2D(128, 256, padding)
        self.dec4 = DownBlock2D(256, 512, padding)

        # Introduce bottleneck here

        self.bottleneck = ResConvBlock2D(512, 1024, 1)

        # Up-conv blocks first upscale, then copy + crop, then conv
        self.up1 = UpBlock2D(1024, 512, padding)
        self.up2 = UpBlock2D(512, 256, padding)
        self.up3 = UpBlock2D(256, 128, padding)
        self.up4 = UpBlock2D(128, 64, padding)

        self.out = nn.Conv2d(64, 4, [1,1])



    def forward(self, x):
        x, skip1 = self.dec1(x)
        x, skip2 = self.dec2(x)
        x, skip3 = self.dec3(x)
        x, skip4 = self.dec4(x)

        x = self.bottleneck(x)

        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)
        x = self.out(x)

        return x

# model = UNet2D(True)

# summary(model, (1,512,512))