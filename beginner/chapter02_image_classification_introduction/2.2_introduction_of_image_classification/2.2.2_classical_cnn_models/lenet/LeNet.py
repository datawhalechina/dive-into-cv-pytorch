# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:44:39 2020

LeNet

LeNet是Yann LeCun等人提出的
最早的卷积神经网络架构之一
多个卷积池化结构堆积+全连接的这种结构设计为后续的神经网络结构研究奠定了基础

本脚本主要练习两点：
lenet 网络搭建
运用lenet进行图像识别-fashion-mnist数据集

@author: 伯禹教育
@modified by: as
"""
#import
import sys
sys.path.append("../../")
import d2lzh_pytorch as d2l
import torch
import torch.nn as nn
import torch.optim as optim
import time 


print('-----------------------------------------------')
print('定义网络结构')
print('-----------------------------------------------')
#net
class Flatten(torch.nn.Module):  #展平操作
    def forward(self, x):
        return x.view(x.shape[0], -1)

class Reshape(torch.nn.Module): #将图像大小重定型
    def forward(self, x):
        return x.view(-1,1,28,28)      #(B x C x H x W)
    
net = torch.nn.Sequential(     #Lelet                                                  
    Reshape(),
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), # b*1*28*28  => b*6*28*28
    nn.Sigmoid(),                                                       
    nn.AvgPool2d(kernel_size=2, stride=2),                              # b*6*28*28  => b*6*14*14
    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),           # b*6*14*14  => b*16*10*10
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),                              # b*16*10*10 => b*16*5*5
    Flatten(),                                                          # b*16*5*5   => b*400
    nn.Linear(in_features=16*5*5, out_features=120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)

print('构造一个高和宽均为28的单通道数据样本，并逐层进行前向计算来查看每个层的输出形状')
X = torch.randn(size=(1,1,28,28), dtype = torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)




print('-----------------------------------------------')
print('训练')
print('-----------------------------------------------')
print('获取数据,我们仍然使用Fashion-MNIST作为训练数据集')
# 数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(
    batch_size=batch_size, root='../../dataset')
print(len(train_iter))


print('查看gpu设备是否可用')
# This function has been saved in the d2l package for future use
#use GPU
def try_gpu():
    """If GPU is available, return torch.device as cuda:0; else return torch.device as cpu."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device
device = try_gpu()
print(device)



print('实现evaluate_accuracy函数，用于计算模型net在数据集data_iter上的准确率')
#计算准确率
'''
注意和之前的evaluate函数区别: 增加了net.eval()
(1). net.train()
  启用 BatchNormalization 和 Dropout，将BatchNormalization和Dropout置为True
(2). net.eval()
不启用 BatchNormalization 和 Dropout，将BatchNormalization和Dropout置为False
'''
def evaluate_accuracy(data_iter, net, device=torch.device('cpu')):
    """Evaluate accuracy of a model on the given data set."""
    acc_sum, n = torch.tensor([0], dtype=torch.float32, device=device), 0
    for X, y in data_iter:
        # If device is the GPU, copy the data to the GPU.
        X, y = X.to(device), y.to(device)
        net.eval()
        with torch.no_grad():
            y = y.long()
            acc_sum += torch.sum((torch.argmax(net(X), dim=1) == y))  #[[0.2 ,0.4 ,0.5 ,0.6 ,0.8] ,[ 0.1,0.2 ,0.4 ,0.3 ,0.1]] => [ 4 , 2 ]
            n += y.shape[0]   # 计算的数量 += batch_num
    return acc_sum.item()/n


# 定义训练函数，注意和之前的区别：
# net.to(device)  X.to(device)
# net.train()
def train_ch5(net, train_iter, test_iter, criterion, num_epochs, batch_size, device, lr=None):
    """Train and evaluate a model with CPU or GPU."""
    print('training on', device)
    net.to(device)  # 将模型参数初始化到对应的设备(这里为了函数的通用性加的，这里其实调用前已经执行过to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        train_l_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        train_acc_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        n, start = 0, time.time()
        for X, y in train_iter:
            net.train()
            
            optimizer.zero_grad()
            X,y = X.to(device), y.to(device) 
            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                y = y.long()
                train_l_sum += loss.float()  # 累加当前batch的loss
                train_acc_sum += (torch.sum((torch.argmax(y_hat, dim=1) == y))).float()  # 计算当前batch中预测正确的个数
                # 这里可以有多种写法，例如：
                #train_acc_sum += torch.sum((torch.argmax(y_hat, dim=1) == y))
                #train_acc_sum += torch.sum((torch.argmax(y_hat, dim=1) == y).float())
                n += y.shape[0]

            # with torch.no_grad() 作用：
            # 直观理解：表示在该区域内的操作，不会涉及梯度，也不会进行反向传播
            # 实际影响：只进行计算（直接操作数据），不让autograd机制记录计算图
            # 这里不存在对网络权重的修改操作，因此不加也不会报错，但是不推荐

        test_acc = evaluate_accuracy(test_iter, net, device)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum/n, train_acc_sum/n, test_acc,
                 time.time() - start))

# 训练
lr, num_epochs = 0.9, 10

# 使用Xavier随机初始化模型权重
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)


# 使用net.to(device) 将模型参数初始化到对应的设备
print('这里进行一个小实验，证明多次对模型执行to(device)是没有影响的')
import time
t1 = time.time()
net = net.to(device)  
t2 = time.time()
print(t2 - t1)
net = net.to(device)  # 将模型参数初始化到对应的设备
t3 = time.time()
print(t3 - t2)

criterion = nn.CrossEntropyLoss()   #交叉熵描述了两个概率分布之间的距离，交叉熵越小说明两者之间越接近
train_ch5(net, train_iter, test_iter, criterion,num_epochs, batch_size, device, lr)


print('最后测试几张图的预测结果')
for testdata,testlabel in test_iter:
    testdata, testlabe = testdata.to(device), testlabel.to(device)
    break
print(testdata.shape,testlabel.shape)
net.eval()
y_pre = net(testdata)
print(torch.argmax(y_pre, dim=1)[:10])
print(testlabel[:10])
