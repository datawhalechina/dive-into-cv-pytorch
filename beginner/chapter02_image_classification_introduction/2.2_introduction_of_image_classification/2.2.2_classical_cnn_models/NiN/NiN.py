# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:54:28 2020

NiN

LeNet、AlexNet和 VGG：先以由卷积层构成的模块充分抽取空间特征，再以由全连接层构成的模块来输出分类结果。
NiN：串联多个由卷积层和“全连接”层(1*1卷积层来替代全连接的功能)构成的小网络来构建一个深层网络。
最后一层使用 global average pooling 来直接完成预测，及 通道数 等于 分类数

1×1卷积核作用
1.放缩通道数：通过控制卷积核的数量达到通道数的放缩。
2.增加非线性。1×1卷积核的卷积过程相当于全连接层的计算过程，并且还加入了非线性激活函数，从而可以增加网络的非线性。
3.计算参数少

主要差异与贡献:
NiN重复使用由卷积层和代替全连接层的1×1卷积层构成的NiN块来构建深层网络。
NiN去除了容易造成过拟合的全连接输出层，而是将其替换成输出通道数等于标签类别数 的NiN块和全局平均池化层。
NiN的以上设计思想影响了后面一系列卷积神经网络的设计。
TODO:待补充

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


def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU())
    return blk


# 已保存在d2lzh_pytorch
class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


# 定义网络结构
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96, 256, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256, 384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2), 
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
    GlobalAvgPool2d(), 
    # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
    d2l.FlattenLayer())


print('打印网络信息')
print(net)


print('打印 1*1*224*224 经过每个模块后的shape')
X = torch.rand(1, 1, 224, 224)
for name, blk in net.named_children(): 
    X = blk(X)
    print(name, 'output shape: ', X.shape)


batch_size = 16
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)


lr, num_epochs = 0.002, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

