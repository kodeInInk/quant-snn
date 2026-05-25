import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.autograd import Function
import torch
import numpy as np
import matplotlib.pyplot as plt

class ReScaWConv(nn.Module):
    def __init__(self, in_chn, out_chn, num_bits, kernel_size=3, stride=1, padding=1):
        super(ReScaWConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = (kernel_size,kernel_size)
        self.out_channels = out_chn
        # self.bias = None
        # init_act_clip_val = 2.0
        # self.clip_val = torch.Tensor([init_act_clip_val]).cuda()
        # self.zero = torch.Tensor([0]).cuda()
        #bias
        self.num_bits_bias = 8
        self.bias = nn.Parameter(torch.zeros(out_chn), requires_grad=True)
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        #weight
        self.num_bits_weight = num_bits #4
        self.weight = nn.Parameter((torch.rand(self.shape)-0.5) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weight
        #ranges
        w_max = 2 ** (self.num_bits_weight - 1) - 1 #7
        w_min = -w_max - 1 #-8
        b_max = 2 ** (self.num_bits_bias - 1) - 1 #127
        b_min = -b_max - 1 #-128
        #scale/o_channel
        gamma = (2**self.num_bits_weight - 1)/(2**(self.num_bits_weight - 1)) #symmetric quantization
        scale = gamma * torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True) #avg(scaling factor/output channel
        scale = scale.detach() #so that there aint no tracking
        
        #STE int4 weight quantasation
        w_scaled = real_weights / scale
        w_int = torch.clamp(w_scaled * w_max, w_min, w_max)
        w_quant = torch.round(w_int).detach() - w_int.detach() + w_int
         #STE int8 bias quantasation(same per channel)
        b_int = torch.clamp(self.bias / scale.view(self.out_channels) * w_max, b_min, b_max)
        b_quant = torch.round(b_int).detach() - b_int.detach() + b_int
        
        #scaling outside (IP: WEIGHT_ACC + MP_BIAS_ACC)
        y = F.conv2d(x, w_quant, stride=self.stride, padding=self.padding)
        y_c = torch.clamp(y, b_min, b_max) #8bit weight_acc
        y = y_c.detach() - y.detach() + y #STE
        y = y + b_quant.view(1, self.out_channels, 1, 1)
        y_c = torch.clamp(y, b_min, b_max) #8bit MP sturation
        y = y_c.detach() - y.detach() + y #STE
        y = y * (scale / w_max).view(1, self.out_channels, 1, 1) #scale after conv; view: reshapes the tensor, scaling factor gets same number of dimensions as y, filling with 1s all non-o_channel dimensions
        return y


# 8-bit quantization for the first and the last layer
class first_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(first_conv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                         bias)

    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q - self.weight).detach() + self.weight
        return F.conv2d(x, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

class last_fc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(last_fc, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        max = self.weight.data.max()
        weight_q = self.weight.div(max).mul(127).round().div(127).mul(max)
        weight_q = (weight_q - self.weight).detach() + self.weight
        return F.linear(x, weight_q, self.bias)