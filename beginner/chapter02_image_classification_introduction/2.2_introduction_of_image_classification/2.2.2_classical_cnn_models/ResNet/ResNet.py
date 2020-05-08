# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 17:20:25 2020

残差网络 Residual Neural Network

深度学习的问题：深度CNN网络达到一定深度后再一味地增加层数并不能带来进一步地分类性能提高，反而会招致网络收敛变得更慢，准确率也变得更差。

ResNet模型结构：
上来先是简单的特征提取并迅速降低分辨率: 卷积(64,7x7,3) + BN + 最大池化(3x3,2)
blockx4 (除第一个block外，每一个block通过第一个步幅为2的残差块来完成pooling的工作并使通道数翻倍)
全局平均池化
全连接

@author: 伯禹教育
@modified by: as
"""
import os
import sys
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
sys.path.append("../")
import d2lzh_pytorch as d2l

# 设置CUDA可见设备
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


class Residual(nn.Module):  # 本类已保存在d2lzh_pytorch包中方便以后使用
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        """
            use_1×1conv: 是否使用额外的1x1卷积层来修改通道数
            stride: 卷积层的步幅, resnet使用步长为2的卷积来替代pooling的作用，是个很赞的idea
        """
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


print('测试通道数不变的情况')
blk = Residual(3, 3)
X = torch.rand((4, 3, 6, 6))
print(blk(X).shape) # torch.Size([4, 3, 6, 6])


print('测试通道数变化的情况')
blk = Residual(3, 6, use_1x1conv=True, stride=2)
print(blk(X).shape) # torch.Size([4, 6, 3, 3])



def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    '''
    resnet block

    num_residuals: 当前block包含多少个残差块
    first_block: 是否为第一个block

    一个resnet block由num_residuals个残差块组成
    其中第一个残差块起到了通道数的转换和pooling的作用
    后面的若干残差块就是完成正常的特征提取
    '''
    if first_block:
        assert in_channels == out_channels # 第一个模块的输出通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


# 定义resnet模型结构
# 上来先是简单的特征提取并迅速降低分辨率: 卷积(64,7x7,3) + BN + 最大池化(3x3,2)
net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), 
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# 然后是连续4个block
net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
net.add_module("resnet_block2", resnet_block(64, 128, 2))
net.add_module("resnet_block3", resnet_block(128, 256, 2))
net.add_module("resnet_block4", resnet_block(256, 512, 2))
# global average pooling
net.add_module("global_avg_pool", d2l.GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
# fc layer
net.add_module("fc", nn.Sequential(d2l.FlattenLayer(), nn.Linear(512, 10))) 


print('打印网络结构')
print(net)


print('打印 1*1*224*224 输入经过每个模块后的shape')
X = torch.rand((1, 1, 224, 224))
for name, layer in net.named_children():
    X = layer(X)
    print(name, ' output shape:\t', X.shape)


print('训练...')
batch_size = 16  #如训练时出现“out of memory”的报错信息，可减小batch_size或resize
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
