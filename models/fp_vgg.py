import torch.nn as nn
from collections import OrderedDict
from models.layers import *
import math

defaultcfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
relucfg = [1, 4, 6, 9, 11, 13, 16, 18, 20, 23, 25, 27]

class VGG(nn.Module):
    def __init__(self, compress_rate, cfg=None, num_classes=10, step=2, num_bits=None):
        super(VGG, self).__init__()

        self.T = step
        print(self.T)

        if cfg is None:
            cfg = defaultcfg
        self.relucfg = relucfg

        self.compress_rate = compress_rate[:]
        self.compress_rate.append(0.0)

        self.features = self._make_layers(cfg)
        self.avgpool = SeqToANNContainer(nn.AvgPool2d(2))
        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', SeqToANNContainer(nn.Linear(512, num_classes)))]))
        self.layer_outputs = []
        for name, module in self.named_modules():
            if isinstance(module, LIFSpike):
                module.register_forward_hook(lambda m, inp, out: self.layer_outputs.append(out))

    def _make_layers(self, cfg):

        layers = nn.Sequential()
        in_channels = 3
        cnt=0

        for i, x in enumerate(cfg):
            if x == 'M':
                layers.add_module('pool%d' % i, SeqToANNContainer(nn.MaxPool2d(kernel_size=2, stride=2)))
            else:
                x = int(x * (1-self.compress_rate[cnt]))

                cnt+=1
                conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
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

def vgg_16_bn(compress_rate, num_classes, num_bits):
    return VGG(compress_rate=compress_rate, num_bits=num_bits, num_classes=num_classes)

class vggsnn(nn.Module):
    def __init__(self, num_classes=10):
        super(vggsnn, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        #pool = APLayer(2)
        self.features = nn.Sequential(
            Layer(2,64,3,1,1),
            Layer(64,128,3,1,1),
            pool,
            Layer(128,256,3,1,1),
            Layer(256,256,3,1,1),
            pool,
            Layer(256,512,3,1,1),
            Layer(512,512,3,1,1),
            pool,
            Layer(512,512,3,1,1),
            Layer(512,512,3,1,1),
            pool,
        )
        W = int(48/2/2/2/2)
        # self.T = 4
        self.classifier = SeqToANNContainer(nn.Linear(512*W*W, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x