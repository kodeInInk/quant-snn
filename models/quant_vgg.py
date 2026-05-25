import torch
import torch.nn as nn
from collections import OrderedDict
from models.layers import *
from models.quant_function import ReScaWConv
import math

# defaultcfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
# relucfg = [1, 4, 6, 9, 11, 13, 16, 18, 20, 23, 25, 27]
# ablation_relucfg = [1, 3, 6, 8, 11, 13, 15, 18, 20, 22, 25, 27]


class VGG(nn.Module):
    def __init__(self, compress_rate, num_bits, num_classes, cfg=None, step=4):
        super(VGG, self).__init__()

        self.T = step
        print(self.T)

        if cfg is None:
            cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        self.relucfg = [1, 4, 6, 9, 11, 13, 16, 18, 20, 23, 25, 27]
        self.ablation_relucfg = [1, 3, 6, 8, 11, 13, 15, 18, 20, 22, 25, 27]

        self.compress_rate = compress_rate[:]
        self.compress_rate.append(0.0)
        self.num_bits = num_bits

        self.features = self._make_layers(cfg)
        self.avgpool = SeqToANNContainer(nn.AvgPool2d(2))

        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', SeqToANNContainer(nn.Linear(512, num_classes)))]))

        self._initialize_weights()

        self.layer_outputs = []
        for name, module in self.named_modules():
            if isinstance(module, LIFSpike):
                module.register_forward_hook(lambda m, inp, out: self.layer_outputs.append(out))


    def _make_layers(self, cfg):

        layers = nn.Sequential()
        in_channels = 3
        cnt=0
        x = int(cfg[0] * (1-self.compress_rate[cnt]))
        conv2d = nn.Conv2d(in_channels=in_channels, out_channels=x, kernel_size=3, stride=1, padding=1)
        bn = tdBatchNorm(x)
        layers.add_module('convbn%d' % 0, tdLayer(conv2d, bn))
        layers.add_module('relu%d' % 0, LIFSpike())

        in_channels = x
        for i, x in enumerate(cfg):
            if i==0:
                continue
            else:
                if x == 'M':
                    layers.add_module('pool%d' % i, SeqToANNContainer(nn.MaxPool2d(kernel_size=2, stride=2)))
                else:
                    cnt += 1
                    x = int(x * (1-self.compress_rate[cnt]))
                    conv2d = ReScaWConv(in_chn=in_channels, out_chn=x, num_bits=self.num_bits,
                                              kernel_size=3, stride=1, padding=1)
                    bn = tdBatchNorm(x)
                    layers.add_module('convbn%d' % i, tdLayer(conv2d,bn))
                    layers.add_module('relu%d' % i, LIFSpike())
                    in_channels = x

        return layers

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, ReScaWConv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        self.layer_outputs = []
        x = add_dimention(x, self.T)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x

def vgg_16_bn(compress_rate, num_bits, num_classes):

    return VGG(compress_rate=compress_rate, num_bits=num_bits, num_classes=num_classes)

class vggsnn(nn.Module):
    def __init__(self, compress_rate, num_bits, num_classes, cfg=None):
        super(vggsnn, self).__init__()

        if cfg is None:
            cfg = [64, 128, 'M', 256, 256, 'M', 512, 512,  'M', 512, 512]
        self.relucfg = [1, 4, 6, 9, 11, 14, 16]
        self.ablation_relucfg = [1, 3, 6, 8, 11, 13, 16]

        self.compress_rate = compress_rate[:]
        self.compress_rate.append(0.0)
        self.num_bits = num_bits

        self.features = self._make_layers(cfg)
        self.avgpool = SeqToANNContainer(nn.AvgPool2d(2))

        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', SeqToANNContainer(nn.Linear(4608, num_classes)))]))

        self._initialize_weights()

    def _make_layers(self, cfg):

        layers = nn.Sequential()
        in_channels = 2
        cnt=0
        x = int(cfg[0] * (1-self.compress_rate[cnt]))
        conv2d = nn.Conv2d(in_channels=in_channels, out_channels=x, kernel_size=3, stride=1, padding=1)
        bn = tdBatchNorm(x)
        layers.add_module('convbn%d' % 0, tdLayer(conv2d, bn))
        layers.add_module('relu%d' % 0, LIFSpike())

        in_channels = x
        for i, x in enumerate(cfg):
            if i==0:
                continue
            else:
                if x == 'M':
                    layers.add_module('pool%d' % i, SeqToANNContainer(nn.MaxPool2d(kernel_size=2, stride=2)))
                else:
                    cnt += 1
                    x = int(x * (1-self.compress_rate[cnt]))
                    conv2d = ReScaWConv(in_chn=in_channels, out_chn=x, num_bits=self.num_bits,
                                              kernel_size=3, stride=1, padding=1)
                    bn = tdBatchNorm(x)
                    layers.add_module('convbn%d' % i, tdLayer(conv2d,bn))
                    layers.add_module('relu%d' % i, LIFSpike())
                    in_channels = x

        return layers

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, ReScaWConv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        # x = add_dimention(x, 10)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x