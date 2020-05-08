# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 23:58:19 2020

fine tuning: 模型微调练习

运行此代码，你需要先：
下载hotdog热狗数据集，放置于dataset目录下
下载resnet18的预训练权重文件，放置于pretrain/resnet18下

hotdog下载链接：https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/hotdog.zip
resnet18预训练权重下载链接：https://download.pytorch.org/models/resnet18-5c106cde.pth 

@modified by: as
"""
import os
import sys
sys.path.append("../")
import d2lzh_pytorch as d2l

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models


os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # TODO:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print("进行数据集和预训练权重文件检查")
pretrain_dir = '../pretrain/resnet18'
resnet18_weight_path = os.path.join(pretrain_dir, 'resnet18-5c106cde.pth')
if not os.path.exists(resnet18_weight_path):
    print("预训练权重文件不存在{}".format(resnet18_weight_path))
    raise RuntimeError("please check first")

dataset_dir = '../dataset/hotdog'
if not os.path.exists(dataset_dir):
    print("数据集文件不存在{}".format(resnet18_weight_path))
    raise RuntimeError("please check first")


train_imgs = ImageFolder(os.path.join(dataset_dir, 'train'))
test_imgs = ImageFolder(os.path.join(dataset_dir, 'test'))

print(train_imgs[0])
print(train_imgs[-1])
print("可以看到train_imgs[i]拿到的是一个元组(PIL Image, label)")


# 由于ImageFolder是按顺序的，所以我们可以这样获取正负样本
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]


# 由于我这边是在服务器上运行，可视化的过程就略去了
#d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)


print("分析下数据集图片的shape")
for img in hotdogs:
    print(img.size)
print("可以看到数据集图片尺寸长宽比不一")


# 定义预处理和数据增强
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

test_augs = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize
    ])


# 加载官方resnet18实现
pretrained_net = models.resnet18(pretrained=False)
pretrained_net.load_state_dict(torch.load(resnet18_weight_path))


print('打印原始预训练模型的全连接层')
print(pretrained_net.fc)


print('重新定义全连接层')
pretrained_net.fc = nn.Linear(512, 2)
print(pretrained_net.fc)


# 获取全连接层权重对应的内存id
output_params = list(map(id, pretrained_net.fc.parameters()))
# 获取除全连接层以外的模型参数
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())

lr = 0.01
# 将随机初始化的fc layer学习率设为已经预训练过的部分的10倍
optimizer = optim.SGD([{'params': feature_params},
                       {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                       lr=lr, weight_decay=0.001)


def train_fine_tuning(net, optimizer, batch_size=128, num_epochs=5):
    train_iter = DataLoader(ImageFolder(os.path.join(dataset_dir, 'train'), transform=train_augs),
                            batch_size, shuffle=True, num_workers=2)
    test_iter = DataLoader(ImageFolder(os.path.join(dataset_dir, 'test'), transform=test_augs),
                           batch_size, num_workers=2)
    loss = torch.nn.CrossEntropyLoss()
    d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)


print('-----------------------------')
print('训练：微调模型')
train_fine_tuning(pretrained_net, optimizer, batch_size=160, num_epochs=5)


print('-----------------------------')
print('训练：从头训练')
scratch_net = models.resnet18(pretrained=False, num_classes=2)
lr = 0.1
optimizer = optim.SGD(scratch_net.parameters(), lr=lr, weight_decay=0.001)
train_fine_tuning(scratch_net, optimizer)
