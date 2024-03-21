import torch
from torch import nn

from 网络修改.models.block import Conv, C2f, SPPF, Concat, Pose

class YOLOv8Backbone(nn.Module):
    def __init__(self, base_channels, layers):
        super().__init__()
        # 定义backbone的层
        self.layer1 = Conv(3, base_channels, 3, 2)
        self.layer2 = Conv(base_channels, base_channels * 2, 3, 2)
        self.layer3 = C2f(base_channels * 2, base_channels * 2, 3)
        self.layer4 = Conv(base_channels * 2, base_channels * 4, 3, 2)
        self.layer5 = C2f(base_channels * 4, base_channels * 4, 6)  # P3
        self.layer6 = Conv(base_channels * 4, base_channels * 8, 3, 2)
        self.layer7 = C2f(base_channels * 8, base_channels * 8, 6)  # P4
        self.layer8 = Conv(base_channels * 8, base_channels * 16, 3, 2)
        self.layer9 = C2f(base_channels * 16, base_channels * 16, 3)
        self.layer10 = SPPF(base_channels * 16, base_channels * 16, 5)  # P5

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x5 = self.layer5(x)  # P3 output
        x = self.layer6(x5)
        x7 = self.layer7(x)  # P4 output
        x = self.layer8(x7)
        x = self.layer9(x)
        x10 = self.layer10(x)  # P5 output
        return x5, x7, x10

class YOLOv8Head(nn.Module):
    def __init__(self, nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024)):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.concat = Concat()

        # Layers as per the head configuration
        # Adjust the channels and other parameters based on your network design
        self.c2f_12 = C2f(ch[1], ch[1], 3)
        self.c2f_15 = C2f(ch[0], ch[0], 3)
        self.c2f_18 = C2f(ch[1], ch[1], 3)
        self.c2f_21 = C2f(ch[2], ch[2], 3)

        self.conv_16 = Conv(ch[0], ch[0], 3, 2)
        self.conv_19 = Conv(ch[1], ch[1], 3, 2)

        self.pose = Pose(nc=nc, kpt_shape=kpt_shape, ch=ch)

    def forward(self, features):
        # Assuming features is a list of feature maps from the backbone
        p3, p4, p5 = features  # Extracted feature maps from the backbone

        # Upsample and concatenate as per the head configuration
        x = self.upsample(p5)
        x = self.concat([x, p4])
        x12 = self.c2f_12(x)

        x = self.upsample(x12)
        x = self.concat([x, p3])
        x15 = self.c2f_15(x)

        x = self.conv_16(x15)
        x = self.concat([x, x12])
        x18 = self.c2f_18(x)

        x = self.conv_19(x18)
        x = self.concat([x, p5])
        x21 = self.c2f_21(x)

        # Pose head
        pose_output = self.pose([x15, x18, x21])

        return pose_output

