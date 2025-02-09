import math
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import math

__all__ = ['PreResNet20', 'PreResNet20Drop', 'PreResNet56', 'PreResNet56Drop', 'PreResNet110', 'PreResNet110Drop', 'PreResNet164', 'PreResNet164Drop', 'PreResNet']

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0.0):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity()

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class PreResNet(nn.Module):
    def __init__(self, num_classes=10, depth=110, dropout_rate=0.0):
        super(PreResNet, self).__init__()
        assert (depth - 2) % 6 == 0, "depth should be 6n+2"
        n = (depth - 2) // 6

        block = Bottleneck if depth >= 44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, n, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(
            block, 32, n, stride=2, dropout_rate=dropout_rate
        )
        self.layer3 = self._make_layer(
            block, 64, n, stride=2, dropout_rate=dropout_rate
        )
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dropout_rate=0.0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity(),
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
            )

        layers = list()
        layers.append(
            block(self.inplanes, planes, stride, downsample, dropout_rate=dropout_rate)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout_rate=dropout_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = self.fc(x)

        return x


class PreResNet20(PreResNet):
    def __init__(self, num_classes=10):
        super(PreResNet20, self).__init__(
            num_classes=num_classes, depth=20, dropout_rate=0
        )


class PreResNet20Drop(PreResNet):
    def __init__(self, num_classes=10):
        super(PreResNet20Drop, self).__init__(
            num_classes=num_classes, depth=20, dropout_rate=0.01
        )


class PreResNet56(PreResNet):
    def __init__(self, num_classes=10):
        super(PreResNet56, self).__init__(
            num_classes=num_classes, depth=56, dropout_rate=0
        )


class PreResNet56Drop(PreResNet):
    def __init__(self, num_classes=10):
        super(PreResNet56Drop, self).__init__(
            num_classes=num_classes, depth=56, dropout_rate=0.01
        )


class PreResNet110(PreResNet):
    def __init__(self, num_classes=10):
        super(PreResNet110, self).__init__(
            num_classes=num_classes, depth=110, dropout_rate=0
        )


class PreResNet110Drop(PreResNet):
    def __init__(self, num_classes=10):
        super(PreResNet110Drop, self).__init__(
            num_classes=num_classes, depth=110, dropout_rate=0.01
        )


class PreResNet164(PreResNet):
    def __init__(self, num_classes=10):
        super(PreResNet164, self).__init__(
            num_classes=num_classes, depth=164, dropout_rate=0
        )


class PreResNet164Drop(PreResNet):
    def __init__(self, num_classes=10):
        super(PreResNet164Drop, self).__init__(
            num_classes=num_classes, depth=164, dropout_rate=0.01)