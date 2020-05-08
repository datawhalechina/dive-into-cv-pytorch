# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:53:59 2020

批量归一化（BatchNormalization）

本小节通过手动实现BN层，来学习其原理

借鉴机器学习模型中对输入进行的z-scoer标准化
(处理后的任意一个特征在数据集中所有样本上的均值为0、标准差为1。)

标准化处理输入数据使各个特征的分布相近，尽可能消除量纲和数据波动带来的影响。
从而能让模型cover住更多情况，获得性能提升。

然而对于深度模型来说，仅仅是对输入进行标准化是不够的。

批量归一化的具体做法是：
利用小批量上的均值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。

BN对于全连接层和卷积层的处理方式稍有不同

BN对全连接层的每个输入节点按batch进行归一化，即：
假设输入维度 batch * N, 归一化参数 均值和方差的维度为 1*N

BN对卷积层以channel为单位按batch进行归一化，即：
假设输入维度 batch * channel * H * W, 归一化参数 均值和方差的维度为 1 * channel * 1 * 1
也就是每个channel共享一组归一化参数

我们这里的实现将BN层放在了激活函数的前面，这对使用sigmoid作为激活函数时显然是有意义的
后续有人指出(包括BN的一个作者)BN放在RELU后面效果更好
可以参考这个回答 https://www.zhihu.com/question/283715823/answer/438882036

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


# 定义一次batch normalization运算的计算图
def batch_norm(is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 判断当前模式是训练模式还是预测模式
    if not is_training:
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算每个通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        # 训练模式下用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 拉伸和偏移
    return Y, moving_mean, moving_var


# 手动实现版本BatchNormalization层的完整定义
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features) #全连接层输出神经元
        else:
            shape = (1, num_features, 1, 1)  #通道数
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 不参与求梯度和迭代的变量，全在内存上初始化成0
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var, Module实例的traning属性默认为true, 调用.eval()后设成false
        Y, self.moving_mean, self.moving_var = batch_norm(self.training, 
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y


# 将BN层应用到LeNet中
net = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            BatchNorm(6, num_dims=4),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            BatchNorm(16, num_dims=4),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            d2l.FlattenLayer(),
            nn.Linear(16*4*4, 120),
            BatchNorm(120, num_dims=2),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            BatchNorm(84, num_dims=2),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

print('打印网络结构')
print(net)


print('打印 1*1*28*28 输入经过每个模块后的shape')
X = torch.rand(1, 1, 28, 28)
for name, blk in net.named_children():
    X = blk(X)
    print(name, 'output shape: ', X.shape)


print('训练...')
batch_size = 16  #如训练时出现“out of memory”的报错信息，可减小batch_size或resize
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

lr, num_epochs = 0.001, 3
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)


print('使用官方实现nn.BatchNorm2d & nn.BatchNorm1d来实现LeNet')
net = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(6),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            d2l.FlattenLayer(),
            nn.Linear(16*4*4, 120),
            nn.BatchNorm1d(120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
print('训练...')
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
