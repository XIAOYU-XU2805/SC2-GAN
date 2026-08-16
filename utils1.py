
#!/usr/bin/python
#-*-coding:utf-8-*-

import pprint
import numpy as np
from glob import glob
import os
import scipy.io as io
import numpy as np
import torch.nn as nn
import math
import scipy.io
import scipy.misc
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import math
from torch.autograd import Variable
import functools
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


pp = pprint.PrettyPrinter()

def get_signal(signal_path):
    data = np.loadtxt(signal_path)
    # data = data[0:5120]
    data_max = np.max(data)
    data_min = np.min(data)
    return data,data_max,data_min

def get_signal_mat(signal_path):
    data = scipy.io.loadmat(signal_path)
    # print(data.keys())
    count = 0
    for key in data.items():
        if(count == 3):
            data = data[key[0]]
        count = count + 1

    # data = data[0:5120]
    data_max = np.max(data)
    data_min = np.min(data)
    return data,data_max,data_min

def get_signal_single(signal_path):
    data = np.loadtxt(signal_path)
    data = data[0:16384]
    data_max = np.max(data)
    data_min = np.min(data)
    return data,data_max,data_min

def get_signal_single_mat(signal_path):
    data = scipy.io.loadmat(signal_path)
    # print(data.keys())
    count = 0
    for key in data.items():
        if(count == 3):
            data = data[key[0]]
        count = count + 1
    data_max = np.max(data)
    data_min = np.min(data)
    return data,data_max,data_min

def double_channel(r_path,i_path,is_norm):
    data_r, r_max, r_min = get_signal_mat(r_path)
    data_i, i_max, i_min = get_signal_mat(i_path)
    data_r = np.expand_dims(data_r, axis=2)
    data_i = np.expand_dims(data_i, axis=2)
    if is_norm:
        data_r = 2 * (data_r - r_min) / abs(r_max - r_min) + (-1)  # 归一化到-1,1区间
        data_i = 2 * (data_i - i_min) / abs(i_max - i_min) + (-1)
    double_channel_data = np.concatenate([data_r, data_i],axis = 2)
    norm = [r_max, r_min, i_max, i_min]
    return [double_channel_data, norm]


def single_channel(r_path,is_norm):
    data_r, r_max, r_min = get_signal_single_mat(r_path)
    if is_norm:
        data_r = 2 * (data_r - r_min) / abs(r_max - r_min) + (-1)  # 归一化到-1,1区间
    double_channel_data = np.expand_dims(data_r, axis=2)
    norm = [r_max, r_min]
    return [double_channel_data, norm]

def data_abs(data):
    data_real = data[:,0]
    data_imag = data[:,1]
    # data_out = data_real
    data_out = np.zeros(data_real.shape)
    # print('data_real.shape={}'.format(data_real.shape[0]))
    for i in range(data_real.shape[0]):
        data_out[i]=math.sqrt(math.pow(data_real[i],2)+math.pow(data_imag[i],2))
    return data_out

def data_abs_2d(data):
    data_real = data[:,:,:,0]
    data_imag = data[:,:,:,1]
    # data_out = data_real
    data_out = np.zeros(data_real.shape)
    # print('data_real.shape={}'.format(data_real.shape[0]))
    # data_out = math.sqrt(math.pow(data_real, 2) + math.pow(data_imag, 2))
    data_out=np.sqrt(np.power(data_real,2)+np.power(data_imag,2))
    # for i in range(data_real.shape[0]):
    #     for j in range(data_real.shape[1]):
    #         for z in range(data_real.shape[2]):
    #             data_out[i,j,z]=math.sqrt(math.pow(data_real[i,j,z],2)+math.pow(data_imag[i,j,z],2))
    return data_out




# def save_signal(signal,signal_path,mat_name):
#     return io.savemat(signal_path,mdict={mat_name:signal})

def save_list(list_name,path):
    file = open(path,'w')
    file.write(str(list_name))
    file.close()
    return

def save_txt(path,data_name):
    np.savetxt(path,data_name)
    return

"""
CondConv
"""
class _routing(nn.Module):
    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        x = torch.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return torch.sigmoid(x)

class CondConv2D(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', num_experts=3, dropout_rate=0.2):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(CondConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        self._routing_fn = _routing(in_channels, num_experts, dropout_rate)

        self.weight = Parameter(torch.Tensor(
        num_experts, out_channels, in_channels // groups, *kernel_size))

        self.reset_parameters()

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

    def forward(self, inputs):
        b, _, _, _ = inputs.size()
        res = []
        for input in inputs:
            input = input.unsqueeze(0)
            pooled_inputs = self._avg_pooling(input)    # GlobalAveragePool(x)
            routing_weights = self._routing_fn(pooled_inputs)   # _routing_fn = Sigmoid(pooled_inputs)*R--->得到r(x)，也就是alpha：样本依赖路由权重
            kernels = torch.sum(routing_weights[:, None, None, None, None] * self.weight, 0)
            out = self._conv_forward(input, kernels)
            res.append(out)
        return torch.cat(res, dim=0)


"""
CBAM
"""
# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_Channel):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.reduction = 16
        # shared MLP

        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(in_Channel, in_Channel // self.reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(in_Channel // self.reduction, in_Channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        out = self.sigmoid(avg_out + max_out)
        out = out.expand_as(x)
        channel_out = out * x
        return channel_out

# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size,padding=kernel_size // 2,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out,max_out], dim=1)
        out = self.conv1(cat)
        out = self.sigmoid(out)
        spatial_out = out * x
        return spatial_out

class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.channel_weight = ChannelAttention(in_channel)
        self.spatial_weight = SpatialAttention(kernel_size=7)
    def forward(self,x):
        F1 = self.channel_weight(x)
        F2 = self.spatial_weight(F1)
        return F2

