# BaseLine

我们目标是训FashionMNIST数据集，在测试集上取得更好的效果

下面我将自己的baseline思路简述如下：

### 分析数据集

首先确认了数据集的图片shape均为：1*28*28

大多数数据集的图片尺寸是不一的，但FashionMNIST是固定的，这省下了很多清洗的工作

### 选择一个模型作为baseline

根据数据集的这样一个规模以及分辨率，肯定不适合用大模型，层数打算控制在20以内

所以我打算尝试一下resnet18作为baseline

我们先把官方的resnet18导入，并print下网络结构

```python
import torch
from torchvision import models

pretrained_net = models.resnet18(pretrained=False)

print('打印网络结构(主要是为了确认如何调整)')
print(pretrained_net)
```

网络结构如下：
```
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=1000, bias=True)
)
```

通过观察，这样的网络结构显然不能直接用。我们模拟下FashionMNIST的输入，看看会发生什么

```python
print('打印 1*1*28*28 输入经过每个模块后的shape')
X = torch.rand((1, 1, 28, 28))
for name, layer in pretrained_net.named_children():
    X = layer(X)
    print(name, ' output shape:\t', X.shape)
```

运行会有如下结果：

```
打印 1*1*28*28 输入经过每个模块后的shape
Traceback (most recent call last):
  File "analysis.py", line 32, in <module>
    X = layer(X)
  File "/data/ansheng/anaconda3/envs/torch_py3_gpu/lib/python3.7/site-packages/torch/nn/modules/module.py", line 541, in __call__
    result = self.forward(*input, **kwargs)
  File "/data/ansheng/anaconda3/envs/torch_py3_gpu/lib/python3.7/site-packages/torch/nn/modules/conv.py", line 345, in forward
    return self.conv2d_forward(input, self.weight)
  File "/data/ansheng/anaconda3/envs/torch_py3_gpu/lib/python3.7/site-packages/torch/nn/modules/conv.py", line 342, in conv2d_forward
    self.padding, self.dilation, self.groups)
RuntimeError: Given groups=1, weight of size 64 3 7 7, expected input[1, 1, 28, 28] to have 3 channels, but got 1 channels instead
```

因此我们第一个需要解决的问题是，官方resnet实现默认的输入是RGB3通道彩色图像，而FashionMNIST是灰度图

调整一下代码，看看还有什么问题

```python
print('打印 1*3*28*28 输入经过每个模块后的shape')
X = torch.rand((1, 3, 28, 28))
for name, layer in pretrained_net.named_children():
    X = layer(X)
    print(name, ' output shape:\t', X.shape)
```

运行结果如下：

```
打印 1*3*28*28 输入经过每个模块后的shape
conv1  output shape:     torch.Size([1, 64, 14, 14])
bn1  output shape:       torch.Size([1, 64, 14, 14])
relu  output shape:      torch.Size([1, 64, 14, 14])
maxpool  output shape:   torch.Size([1, 64, 7, 7])
layer1  output shape:    torch.Size([1, 64, 7, 7])
layer2  output shape:    torch.Size([1, 128, 4, 4])
layer3  output shape:    torch.Size([1, 256, 2, 2])
Traceback (most recent call last):
  File "analysis.py", line 30, in <module>
    X = layer(X)
  File "/data/ansheng/anaconda3/envs/torch_py3_gpu/lib/python3.7/site-packages/torch/nn/modules/module.py", line 541, in __call__
    result = self.forward(*input, **kwargs)
  File "/data/ansheng/anaconda3/envs/torch_py3_gpu/lib/python3.7/site-packages/torch/nn/modules/container.py", line 92, in forward
    input = module(input)
  File "/data/ansheng/anaconda3/envs/torch_py3_gpu/lib/python3.7/site-packages/torch/nn/modules/module.py", line 541, in __call__
    result = self.forward(*input, **kwargs)
  File "/data/ansheng/anaconda3/envs/torch_py3_gpu/lib/python3.7/site-packages/torchvision/models/resnet.py", line 60, in forward
    out = self.bn1(out)
  File "/data/ansheng/anaconda3/envs/torch_py3_gpu/lib/python3.7/site-packages/torch/nn/modules/module.py", line 541, in __call__
    result = self.forward(*input, **kwargs)
  File "/data/ansheng/anaconda3/envs/torch_py3_gpu/lib/python3.7/site-packages/torch/nn/modules/batchnorm.py", line 81, in forward
    exponential_average_factor, self.eps)
  File "/data/ansheng/anaconda3/envs/torch_py3_gpu/lib/python3.7/site-packages/torch/nn/functional.py", line 1666, in batch_norm
    raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))
ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 512, 1, 1])
```

可以看出，resnet18进行的下采样太多了，导致运行到resnet block 4就进行不下去了

这是由于官方的resnet18是在输入图片224*224的尺寸下设计的

因此下采样过多，感受野过大的问题我们也需要解决，因为我们的输入只有28*28

因此我对官方的resnet18进行了以下几点改动：

1. 对第一个卷积核进行比较大的改动，缩小感受野，尺寸3*3，stride=1, input_channel=1
2. 去掉第一个maxpooling
3. 将所有的卷积核的数量减半
4. 修改输出层fc layer的输出类别个数

由于做了比较多的改动，预训练模型肯定是加载不了了。

简单的完成训练部分的代码，运行，可以得到一个0.922分左右的基础baseline：

```
训练...
training on  cuda:0
epoch 1, loss 0.5634, train acc 0.791, test acc 0.844, time 30.7 sec
find best! save at model/best.pth
epoch 2, loss 0.3262, train acc 0.881, test acc 0.871, time 27.9 sec
find best! save at model/best.pth
epoch 3, loss 0.2727, train acc 0.900, test acc 0.872, time 28.4 sec
find best! save at model/best.pth
epoch 4, loss 0.2414, train acc 0.913, test acc 0.907, time 28.3 sec
find best! save at model/best.pth
epoch 5, loss 0.2161, train acc 0.921, test acc 0.906, time 28.7 sec
epoch 6, loss 0.1997, train acc 0.926, test acc 0.886, time 27.9 sec
epoch 7, loss 0.1830, train acc 0.933, test acc 0.917, time 28.8 sec
find best! save at model/best.pth
epoch 8, loss 0.1690, train acc 0.938, test acc 0.914, time 27.5 sec
epoch 9, loss 0.1536, train acc 0.944, test acc 0.920, time 28.8 sec
find best! save at model/best.pth
epoch 10, loss 0.1386, train acc 0.949, test acc 0.922, time 28.4 sec
find best! save at model/best.pth
加载最优模型
inference测试集
生成提交结果文件
```

仅供刚入门的小伙伴参考，欢迎把进一步提分的技巧和喜悦分享给我~
