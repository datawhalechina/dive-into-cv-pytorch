# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 21:19:22 2020

卷积神经网络基础

本节课的视频介绍了卷积神经网络的基础概念，主要是卷积层和池化层，并解释填充、步幅、输入通道和输出通道的含义。

@author: 伯禹教育
@modified by: as
"""

print('------------------------------------')
print('手动实现一个互相关运算')
print('------------------------------------')

import torch 
import torch.nn as nn

def corr2d(X, K):
    H, W = X.shape
    h, w = K.shape
    Y = torch.zeros(H - h + 1, W - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
Y = corr2d(X, K)
print(Y)


print('------------------------------------')
print('构造一个实验来让卷积核学会如何识别边缘')
print('------------------------------------')
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
# nn.Parameter用于将一个不可训练的类型Tensor转换成可以训练的类型parameter
# 并将这个parameter绑定到这个module的net.parameter()中

X = torch.ones(6, 8)
Y = torch.zeros(6, 7)
X[:, 2: 6] = 0
Y[:, 1] = 1
Y[:, 5] = -1
print('构造具有水平方向变化的输入X和目标输出Y')
print(X)
print(Y)

print('尝试训练一个1*2卷积层来检测颜色边缘')
conv2d = Conv2D(kernel_size=(1, 2))
step = 30
lr = 0.01
for i in range(step):
    Y_hat = conv2d(X)
    l = ((Y_hat - Y) ** 2).sum()  # 这里相当于用均方误差作为loss函数
    l.backward()
    # 梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad
    
    # 梯度清零
    conv2d.weight.grad.zero_()
    conv2d.bias.grad.zero_()
    if (i + 1) % 5 == 0:
        print('Step %d, loss %.3f' % (i + 1, l.item()))
        
print(conv2d.weight.data)
print(conv2d.bias.data)


print('尝试不用 conv2d.weight.data 的方式训练')
conv2d = Conv2D(kernel_size=(1, 2))
step = 30
lr = 0.01
for i in range(step):
    Y_hat = conv2d(X)
    l = ((Y_hat - Y) ** 2).sum()  # 这里相当于用均方误差作为loss函数
    l.backward()
    # 梯度下降
    #conv2d.weight -= lr * conv2d.weight.grad
    #conv2d.bias -= lr * conv2d.bias.grad
    # 上面两行会报错：RuntimeError: a leaf Variable that requires grad has been used in an in-place operation.
    #weight_grad = conv2d.weight.grad.clone()
    #conv2d.weight = conv2d.weight - lr * weight_grad 
    #conv2d.bias = conv2d.bias - lr * nn.Parameter(conv2d.bias.grad)
    #conv2d.weight = conv2d.weight - lr * conv2d.weight.grad
    with torch.no_grad():
        conv2d.weight -= lr * conv2d.weight.grad
        conv2d.bias -= lr * conv2d.bias.grad
    #conv2d.weight.detach() -= lr * conv2d.weight.grad
    #conv2d.bias.detach() -= lr * conv2d.bias.grad
    
    # 梯度清零
    conv2d.weight.grad.zero_()
    conv2d.bias.grad.zero_()
    if (i + 1) % 5 == 0:
        print('Step %d, loss %.3f' % (i + 1, l.item()))
        
print(conv2d.weight.data)
print(conv2d.bias.data)



print('------------------------------------')
print('卷积层的pytorch官方实现')
print('------------------------------------')
# 卷积层官方函数 nn.Conv2d
# in_channels (python:int) – Number of channels in the input image
# out_channels (python:int) – Number of channels produced by the convolution
# kernel_size (python:int or tuple) – Size of the convolving kernel
# stride (python:int or tuple, optional) – Stride of the convolution. Default: 1
# padding (python:int or tuple, optional) – Zero-padding added to both sides of the input. Default: 0
# bias (bool, optional) – If True, adds a learnable bias to the output. Default: True
# forward函数的参数为一个四维张量，形状为NCHW，返回值也是一个四维张量，形状为NCHW,其中NCHW分别表示批量大小、通道数、高度、宽度。

X = torch.rand(4, 2, 3, 5)
print(X.shape)
conv2d = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=(3, 5), stride=1, padding=(1, 2))
Y = conv2d(X)
print('Y.shape: ', Y.shape)
print('weight.shape: ', conv2d.weight.shape)
print('bias.shape: ', conv2d.bias.shape)



print('------------------------------------')
print('池化层的简洁实现')
print('------------------------------------')
# 最大池化层官方函数nn.MaxPool2d
# kernel_size – the size of the window to take a max over
# stride – the stride of the window. Default value is kernel_size
# padding – implicit zero padding to be added on both sides
# forward函数的参数为一个四维张量，形状为NCHW，返回值也是一个四维张量，形状为NCHW,其中NCHW分别表示批量大小、通道数、高度、宽度。
X = torch.arange(32, dtype=torch.float32).view(1, 2, 4, 4)
pool2d = nn.MaxPool2d(kernel_size=3, padding=1, stride=(2, 1))
Y = pool2d(X)
print(X)
print(Y)

