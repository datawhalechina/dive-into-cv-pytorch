# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 22:49:37 2020

VGG11

主要差异与贡献:
TODO:
给出一种可扩展的网络范式，通过重复使用简单的基础模块来构建深度网络
最初的几个卷积层也使用了3*3卷积核

@author: 伯禹教育
@modified by: as
"""
import os
import sys
import time
import torch
from torch import nn, optim
import torchvision
import numpy as np
sys.path.append("../")
import d2lzh_pytorch as d2l
import torch.nn.functional as F


def vgg_block(num_convs, in_channels, out_channels): #卷积层个数，输入通道数，输出通道数
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 这里会使宽高减半
    return nn.Sequential(*blk)


def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # 卷积层部分
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        # 每经过一个vgg_block都会使宽高减半
        net.add_module("vgg_block_" + str(i+1), vgg_block(num_convs, in_channels, out_channels))
    # 全连接层部分
    net.add_module("fc", nn.Sequential(d2l.FlattenLayer(),
                                 nn.Linear(fc_features, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, 10)
                                ))
    return net


conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
# 经过5个vgg_block, 宽高会减半5次, 变成 224/32 = 7
fc_features = 512 * 7 * 7 # c * w * h
fc_hidden_units = 4096 # 任意
net = vgg(conv_arch, fc_features, fc_hidden_units)


print('打印网络信息')
print(net)


print('打印 1*224*224 输入图像经过每个VGG Block之后的尺寸')
X = torch.rand(1, 1, 224, 224)
for name, blk in net.named_children(): 
    # named_children获取一级子模块及其名字(named_modules会返回所有子模块,包括子模块的子模块)
    X = blk(X)
    print(name, 'output shape: ', X.shape)


batch_size = 16  #如训练时出现“out of memory”的报错信息，可减小batch_size或resize
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, 224)
for X, Y in train_iter:
    print('X =', X.shape)
    print('Y =', Y.type(torch.int32))
    break

lr, num_epochs = 0.001, 3
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
