__all__ = ('BasicBlock', 'ResNet1D')

import torch.nn as nn

def resnet34(num_outputs, **kwargs):
    return ResNet1D([3, 4, 6, 3], num_outputs, **kwargs)


def resnet18(num_outputs, **kwargs):
    return ResNet1D([2, 2, 2, 2], num_outputs, **kwargs)


class BottleneckBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1, expansion=None):
        super(BottleneckBlock, self).__init__()
        self.expansion = expansion
        if out_chan // in_chan == 4: # first block
            mid_chan = in_chan
        elif out_chan // in_chan == 2: #first in block sequence
            mid_chan = in_chan // 2
        elif out_chan == in_chan: #not first in block sequence
            mid_chan = in_chan // 4

        self.conv1 = nn.Conv1d(in_chan, mid_chan, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(mid_chan)
        self.conv2 = nn.Conv1d(mid_chan, mid_chan, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(mid_chan)
        self.conv3 = nn.Conv1d(mid_chan, out_chan, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_chan)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_chan != out_chan:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_chan)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += shortcut
        x = self.relu(x)

        return x


class BasicBlock(nn.Module):
    # noinspection PyUnusedLocal
    def __init__(self, in_chan, out_chan, kernel_size, stride=1, expansion=None):
        super().__init__()
        if stride > 1 or in_chan != out_chan:
            self.shortcut = conv1d(in_chan, out_chan, 1, stride)
        else:
            self.shortcut = nn.Identity()
        self.bn1 = nn.BatchNorm1d(in_chan)
        self.relu1 = nn.ReLU()
        self.conv1 = conv1d(in_chan, out_chan, kernel_size, stride)
        self.bn2 = nn.BatchNorm1d(out_chan)
        self.relu2 = nn.ReLU()
        self.conv2 = conv1d(out_chan, out_chan, kernel_size)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        out = x + shortcut
        return out


class ResNet1D(nn.Module): # TODO change to attention pooling and change batch normalization to group normalization
    def __init__(self, stages, num_outputs, block=BasicBlock,
                 in_chan=1, out_chan=64, kernel_size=None, stem_kernel_size=7):
        #stages: number of blocks in each of the 4 modules
        super().__init__()
        if kernel_size is None:
            kernel_size = [3] * len(stages)
        elif isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(stages)
        self.conv1 = conv1d(in_chan, out_chan, stem_kernel_size, stride=2)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.stages = []
        if block == BasicBlock:
            in_chan = out_chan
        elif block == BottleneckBlock:
            in_chan=64
            out_chan = 256
        for i, num_blocks in enumerate(stages):
            stride = 1 if i == 0 else 2 # conv_2 with stride 1, conv_3,4,5  with stride 2
            self.stages.append(
                self._init_stage(block, num_blocks, in_chan,
                                 out_chan, kernel_size[i], stride))
            in_chan = out_chan
            out_chan = 2 * in_chan
        self.stages = nn.Sequential(*self.stages)
        self.bn1 = nn.BatchNorm1d(in_chan)
        self.relu1 = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Linear(in_chan, num_outputs)

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                        nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


    @staticmethod
    def _init_stage(block, num_blocks, in_chan, out_chan, kernel_size, stride):
        blocks = [block(in_chan, out_chan, kernel_size, stride)]
        for i in range(num_blocks - 1):
            blocks.append(block(out_chan, out_chan, kernel_size))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stages(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.avgpool(x).squeeze(-1)
        out = self.fc(x)
        return out


def conv1d(in_chan, out_chan, kernel_size, stride=1):
    assert kernel_size % 2 == 1  # for simplicity
    padding = (kernel_size - 1) // 2
    return nn.Conv1d(in_chan, out_chan, kernel_size,
                     stride, padding, bias=True)


