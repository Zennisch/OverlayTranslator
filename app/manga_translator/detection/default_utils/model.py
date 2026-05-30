import torch
import torch.nn as nn
from torchvision.models import ResNet34_Weights, resnet34


class DBHead(nn.Module):
    def __init__(self, in_channels, k=50):
        super().__init__()
        self.k = k
        self.binarize = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 4, 2, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 4, 2, 1),
        )
        self.binarize.apply(self.weights_init)

        self.thresh = self._init_thresh(in_channels)
        self.thresh.apply(self.weights_init)

    def forward(self, x):
        shrink_maps = self.binarize(x)
        threshold_maps = self.thresh(x)
        if self.training:
            binary_maps = self.step_function(shrink_maps.sigmoid(), threshold_maps)
            y = torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)
        else:
            y = torch.cat((shrink_maps, threshold_maps), dim=1)
        return y

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels, serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels // 4, smooth=smooth, bias=bias),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid(),
        )
        return self.thresh

    def _init_upsample(self, in_channels, out_channels, smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias),
            ]
            if out_channels == 1:
                module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=True))
            return nn.Sequential(*module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


class DoubleConv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stride=1):
        super(DoubleConv, self).__init__()
        self.down = None
        if stride > 1:
            self.down = nn.AvgPool2d(2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.down is not None:
            x = self.down(x)
        x = self.conv(x)
        return x


class DoubleConvUp(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(DoubleConvUp, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class TextDetection(nn.Module):
    def __init__(self, pretrained=None):
        super(TextDetection, self).__init__()
        self.backbone = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)

        self.conv_db = DBHead(64)

        self.conv_mask = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.down_conv1 = DoubleConv(0, 512, 512, 2)
        self.down_conv2 = DoubleConv(0, 512, 512, 2)
        self.down_conv3 = DoubleConv(0, 512, 512, 2)

        self.upconv1 = DoubleConvUp(0, 512, 256)
        self.upconv2 = DoubleConvUp(256, 512, 256)
        self.upconv3 = DoubleConvUp(256, 512, 256)
        self.upconv4 = DoubleConvUp(256, 512, 256)
        self.upconv5 = DoubleConvUp(256, 256, 128)
        self.upconv6 = DoubleConvUp(128, 128, 64)
        self.upconv7 = DoubleConvUp(64, 64, 64)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)  # 64@384

        h4 = self.backbone.layer1(x)  # 64@384
        h8 = self.backbone.layer2(h4)  # 128@192
        h16 = self.backbone.layer3(h8)  # 256@96
        h32 = self.backbone.layer4(h16)  # 512@48
        h64 = self.down_conv1(h32)  # 512@24
        h128 = self.down_conv2(h64)  # 512@12
        h256 = self.down_conv3(h128)  # 512@6

        up256 = self.upconv1(h256)  # 128@12
        up128 = self.upconv2(torch.cat([up256, h128], dim=1))  # 64@24
        up64 = self.upconv3(torch.cat([up128, h64], dim=1))  # 128@48
        up32 = self.upconv4(torch.cat([up64, h32], dim=1))  # 64@96
        up16 = self.upconv5(torch.cat([up32, h16], dim=1))  # 128@192
        up8 = self.upconv6(torch.cat([up16, h8], dim=1))  # 64@384
        up4 = self.upconv7(torch.cat([up8, h4], dim=1))  # 64@768

        return self.conv_db(up8), self.conv_mask(up4)
