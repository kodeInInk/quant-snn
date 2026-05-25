import math
import torch.nn as nn
import torch.nn.functional as F
from models.layers import *
from models.quant_function import ReScaWConv

def adapt_channel(compress_rate, num_layers):
    if num_layers==20:
        stage_repeat = [3, 3, 3]
        stage_out_channel = [64] + [128] * 3 + [256] * 3 + [512] * 3
        # stage_out_channel = [64] + [16] * 3 + [32] * 3 + [64] * 3    # ori
    elif num_layers==32:
        stage_repeat = [5, 5, 5]
        stage_out_channel = [16] + [16] * 5 + [32] * 5 + [64] * 5
    elif num_layers==44:
        stage_repeat = [7, 7, 7]
        stage_out_channel = [16] + [16] * 7 + [32] * 7 + [64] * 7
    elif num_layers==56:
        stage_repeat = [9, 9, 9]
        stage_out_channel = [16] + [16] * 9 + [32] * 9 + [64] * 9
    elif num_layers==110:
        stage_repeat = [18, 18, 18]
        stage_out_channel = [16] + [16] * 18 + [32] * 18 + [64] * 18

    stage_oup_cprate = []
    stage_oup_cprate += [compress_rate[0]]
    for i in range(len(stage_repeat)-1):
        stage_oup_cprate += [compress_rate[i+1]] * stage_repeat[i]
    stage_oup_cprate +=[0.] * stage_repeat[-1]
    mid_cprate = compress_rate[len(stage_repeat):]

    overall_channel = []
    mid_channel = []
    for i in range(len(stage_out_channel)):
        if i == 0 :
            overall_channel += [int(stage_out_channel[i] * (1-stage_oup_cprate[i]))]
        else:
            overall_channel += [int(stage_out_channel[i] * (1-stage_oup_cprate[i]))]
            mid_channel += [int(stage_out_channel[i] * (1-mid_cprate[i-1]))]

    return overall_channel, mid_channel


def conv3x3(in_planes, out_planes, stride=1, num_bit=None):
    """3x3 convolution with padding"""
    return ReScaWConv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, num_bits=num_bit)

def conv1x1(in_planes, out_planes, stride=1, num_bit=None):
    """1x1 convolution"""
    return ReScaWConv(in_planes, out_planes, kernel_size=1, stride=stride, num_bits=num_bit)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = SeqToANNContainer(lambd)

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, midplanes, inplanes, planes, stride=1, num_bit=None):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv3x3(inplanes, midplanes, stride, num_bit)
        self.bn1 = tdBatchNorm(midplanes)
        self.conv1_s = tdLayer(self.conv1, self.bn1)
        # conv1 = conv3x3(inplanes, midplanes, stride, num_bit)
        # bn1 = tdBatchNorm(midplanes)
        # self.conv1_s = tdLayer(conv1, bn1)
        self.relu1 = LIFSpike()

        self.conv2 = conv3x3(midplanes, planes,num_bit=num_bit)
        self.bn2 = tdBatchNorm(planes)
        self.conv2_s = tdLayer(self.conv2, self.bn2)
        # conv2 = conv3x3(midplanes, planes, num_bit=num_bit)
        # bn2 = tdBatchNorm(planes)
        # self.conv2_s = tdLayer(conv2, bn2)
        self.relu2 = LIFSpike()
        self.stride = stride

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            if stride!=1:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2],
                                    (0, 0, 0, 0, (planes-inplanes)//2, planes-inplanes-(planes-inplanes)//2), "constant", 0))
            else:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, :, :],
                                    (0, 0, 0, 0, (planes-inplanes)//2, planes-inplanes-(planes-inplanes)//2), "constant", 0))
            # # self.shortcut = LambdaLayer(
            # #   lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4),"constant", 0))

            '''self.shortcut = nn.Sequential(
                conv1x1(inplanes, planes, stride=stride),
                #nn.BatchNorm2d(planes),
            )#'''

    def forward(self, x):
        out = self.conv1_s(x)
        out = self.relu1(out)

        out = self.conv2_s(out)

        #print(self.stride, self.inplanes, self.planes, out.size(), self.shortcut(x).size())
        out += self.shortcut(x)
        out = self.relu2(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_layers, compress_rate, num_classes, num_bits, step):
        super(ResNet, self).__init__()
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        n = (num_layers - 2) // 6

        self.T = step
        self.num_bits = num_bits

        self.num_layer = num_layers
        self.overall_channel, self.mid_channel = adapt_channel(compress_rate, num_layers)

        self.layer_num = 0

        self.conv1 = nn.Conv2d(3, self.overall_channel[self.layer_num], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = tdBatchNorm(self.overall_channel[self.layer_num])
        self.conv1_s = tdLayer(self.conv1, self.bn1)
        # conv1 = nn.Conv2d(3, self.overall_channel[self.layer_num], kernel_size=3, stride=1, padding=1, bias=False)
        # bn1 = tdBatchNorm(self.overall_channel[self.layer_num])
        # self.conv1_s = tdLayer(conv1, bn1)

        self.relu = LIFSpike()
        self.layers = nn.ModuleList()
        self.layer_num += 1

        #self.layers = nn.ModuleList()
        self.layer1 = self._make_layer(block, blocks_num=n, stride=1)
        self.layer2 = self._make_layer(block, blocks_num=n, stride=2)
        self.layer3 = self._make_layer(block, blocks_num=n, stride=2)

        self.avgpool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = SeqToANNContainer(nn.Linear(512 * BasicBlock.expansion, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m,ReScaWConv):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, blocks_num, stride):
        layers = []
        layers.append(block(self.mid_channel[self.layer_num - 1], self.overall_channel[self.layer_num - 1],
                                 self.overall_channel[self.layer_num], stride, self.num_bits))
        self.layer_num += 1

        for i in range(1, blocks_num):
            layers.append(block(self.mid_channel[self.layer_num - 1], self.overall_channel[self.layer_num - 1],
                                     self.overall_channel[self.layer_num], num_bit=self.num_bits))
            self.layer_num += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = add_dimention(x, self.T)
        x = self.conv1_s(x)
        x = self.relu(x)

        for i, block in enumerate(self.layer1):
            x = block(x)
        for i, block in enumerate(self.layer2):
            x = block(x)
        for i, block in enumerate(self.layer3):
            x = block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.fc(x)
        return x

def resnet_20(compress_rate, num_bits, num_classes):
    # params 0.46M
    T = 2
    return ResNet(BasicBlock, 20, compress_rate=compress_rate, num_bits=num_bits, num_classes=num_classes, step=T)
