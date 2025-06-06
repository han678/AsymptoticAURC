"""
    WideResNet model definition
    ported from https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py
"""

import torch.nn as nn
import torch.nn.functional as F

__all__ = ["WideResNet28x10", "WideResNet28x10Drop"]


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True
    )


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, stride=1, dropout_rate=0):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True)
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    def __init__(self, num_classes=10, depth=28, widen_factor=10, dropout_rate=0):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = (depth - 4) / 6
        k = widen_factor

        nstages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, nstages[0])
        self.layer1 = self._wide_layer(WideBasic, nstages[1], n, stride=1)
        self.layer2 = self._wide_layer(WideBasic, nstages[2], n, stride=2)
        self.layer3 = self._wide_layer(WideBasic, nstages[3], n, stride=2)
        self.bn1 = nn.BatchNorm2d(nstages[3], momentum=0.9)
        self.linear = nn.Linear(nstages[3], num_classes)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity()

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * int(num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)

        return out


class WideResNet28x10(WideResNet):
    def __init__(self, num_classes=10):
        super(WideResNet28x10, self).__init__(
            num_classes=num_classes, depth=28, widen_factor=10, dropout_rate=0
        )


class WideResNet28x10Drop(WideResNet):
    def __init__(self, num_classes=10, dropout_rate=0.05):
        super(WideResNet28x10Drop, self).__init__(
            num_classes=num_classes,
            depth=28,
            widen_factor=10,
            dropout_rate=dropout_rate,
        )