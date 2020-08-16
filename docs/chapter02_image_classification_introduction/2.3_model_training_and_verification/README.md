# 基于Cifar10的图像分类入门学习-PyTorch版

##  图像分类小目标

- 数据预处理、加载
- 模型训练、调参
- 模型保存、加载

我们通过Pytorch来训练一个小分类模型，展示建立分类器的具体步骤：

#### 1 数据预处理、加载

AI数据主要包括：文本、图像、音频、视频数据，这些数据可使用标准Python数据包加载，放到一个numpy数组，讲数组转换为torch.* Tensor。其中：

- 图像数据，常用OpenCV，Pillow包

- 音频数据，常用scipy，librosa包

- 文本数据，常用NLTK, SpaCy包

Pytorch包涵盖常用数据集，可通过torchvision.datasets读取，并使用torchvision加载并预处理CIFAR-10数据集。具体可参考：[Pytorch数据读取方法简介](https://github.com/monkeyDemon/Dive-into-CV-PyTorch/tree/develop/beginner/chapter02_image_classification_introduction/2.1_dataloader_and_augmentation)

本文使用Cifar10数据集，包含10类，分别为： ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’。图像大小均为32x32x3。

<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.3_model_training_and_verification/dataset_ex.png">

```python
import torch
import torchvision
import torchvision.transforms as transforms

# 转化为Tensor，将元素转化为0-1的数字，Normalize将其归一化。
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 训练集需要训练
trainset = torchvision.datasets.CIFAR10('../../../dataset', train=True,transform=None, target_transform=None, download=True)

# batch_size设置了批量大小，shuffle设置为True在装载过程中为随机乱序，num_workers>=1表示多线程读取数据。
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=2)

# 测试集不需要训练
testset = torchvision.datasets.CIFAR10('../../../dataset',train=False,transform=None, target_transform=None, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=2)

# 指定类别标签
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

可查看图像

```python
import matplotlib.pyplot as plt
import numpy as np

def imgshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()
imgshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
```

<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.3_model_training_and_verification/dataset_check.png">

#### 2 模型训练、调参：

2.1 定义一个神经网络（NN）

**知识点**

1. **卷积层**

```
self.conv1 = nn.Conv2d(3,8,3,padding=1)
定义一次卷积运算，其中第一个3表示输入为3通道对应到本次测试为图片的RGB三个通道，数字8的意思为8个卷积核，第二个3表示卷积核的大小为3x3，padding=1表示在图片的周围增加一层像素值用来保存图片的边缘信息。
```
官网参考：
```
class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```
- in_channels(int) ：输入信号的通道
- out_channels(int) ：卷积产生的通道
- kerner_size(int or tuple) ：卷积核的尺寸
- stride(int or tuple, optional) ：卷积步长
- padding(int or tuple, optional) ：输入的每一条边补充0的层数
- dilation(int or tuple, optional) ：卷积核元素之间的间距
- groups(int, optional) ：从输入通道到输出通道的阻塞连接数
- bias(bool, optional) ：如果bias=True，添加偏置


2. **池化层**
```
self.pool1 = nn.MaxPool2d(2,2)
二维池化其中第一个2表示池化窗口的大小为2x2，第二个2表示窗口移动的步长。
```
官网参考：
```
class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
```
- kernel_size(int or tuple) ：max pooling的窗口大小
- stride(int or tuple, optional) ：max pooling的窗口移动的步长。默认值是kernel_size
- padding(int or tuple, optional) ：输入的每一条边补充0的层数
- dilation(int or tuple, optional) ：一个控制窗口中元素步幅的参数
- return_indices ： 如果等于True，会返回输出最大值的序号，对于上采样操作会有帮助
- ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作


3. **归一化**
```
self.bn1 = nn.BatchNorm2d(64)
```
函数介绍：
```
torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
```
- num_features： 来自期望输入的特征数，该期望输入的大小为batch_size x num_features [x width]
- eps： 为保证数值稳定性（分母不能趋近或取0）,给分母加上的值。默认为1e-5。
- momentum： 动态均值和动态方差所使用的动量。默认为0.1。
- affine： 布尔值，当设为true，给该层添加可学习的仿射变换参数。
- track_running_stats：布尔值，当设为true，记录训练过程中的均值和方差；

在进行训练之前，一般要对数据做归一化，使其分布一致，但是在深度神经网络训练过程中，通常以送入网络的每一个batch训练，这样每个batch具有不同的分布；此外，为了解决internal covarivate shift问题，这个问题定义是随着batch normalizaiton这篇论文提出的，在训练过程中，数据分布会发生变化，对下一层网络的学习带来困难。
所以batch normalization就是强行将数据拉回到均值为0，方差为1的正太分布上，这样不仅数据分布一致，而且避免发生梯度消失。

4. **ReLU激活函数**

```
self.relu1 = nn.ReLU()
```
激活函数图像

<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.3_model_training_and_verification/relu.png">

激活函数（Activation Function），是在人工神经网络的神经元上运行的函数，负责将神经元的输入映射到输出端。引入激活函数是为了增加神经网络模型的非线性。没有激活函数的每层都相当于矩阵相乘。就算你叠加了若干层之后，无非还是个矩阵相乘罢了。

如果不用激活函数，每一层输出都是上层输入的线性函数，无论神经网络有多少层，输出都是输入的线性组合，这种情况就是最原始的感知机（Perceptron）。如果使用的话，激活函数给神经元引入了非线性因素，使得神经网络可以任意逼近任何非线性函数，这样神经网络就可以应用到众多的非线性模型中。

Relu激活函数（The Rectified Linear Unit）修正线性单元，用于隐层神经元输出。

5. **全连接层**

```
self.fc14 = nn.Linear(512x4x4,1024)
```
全连层输入数据个数为一维数据512x4x4，输出个数为1024.

6. **Dropout**

```
self.drop1 = nn.Dropout2d()
```
Dropout：删除掉隐藏层随机选取的一半神经元，然后在这个更改的神经元网络上正向和反向更新，然后再恢复之前删除过的神经元，重新选取一般神经元删除，正向反向，更新w,b.重复此过程，最后学习出来的神经网络中的每个神经元都是在一半神经元的基础上学习的，当所有神经元被恢复后，为了补偿，我们把隐藏层的所有权重减半。

为什么Dropout可以减少overfitting？
每次扔掉了一般隐藏层的神经元，相当于在不同的神经网络训练了，减少了神经元的依赖性，迫使神经网络去学习更加健硕的特征。

**Code**
- conv卷积的结果是32x32x18，可以理解为32x32大小的图片共有18张（18通道）
- 卷积操作后relu激活函数增加非线性拟合能力
- maxpool之后结果是16x16x18 ，可以理解为16x16大小的图片共有18张（18通道）
- 全连接层fc1，输入层18x16x16 = 4608个神经元，输出层有64个神经元
- 全连接层fc2，输入层有64个神经元，输出层有10个神经元，对应的是十个标签

```python
from torch.autograd.variable import Variable
import torch.nn.functional as F
 
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, padding=1, stride=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
        self.fc2 = torch.nn.Linear(64, 10)
 
    # 前向传播
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # view将池化后的张量拉伸，-1的意思其实就是未知数的意思，根据其他位置（这里就是18*16*16）来推断这个-1是几
        x = x.view(-1, 18 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```
2.2 定义损失函数及优化

```python
import torch.optim as optim

def createlossandoptimizer(net, learning_rate=0.001):
    loss = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)     # Adam 优化算法是随机梯度下降算法的扩展式
    print(optimizer)
    return loss, optimizer
```
2.3 定义训练、验证、预测模块

```python
def get_train_loader(batch_size):
    # train_loader， 一次性加载了sample中全部的样本数据，每次以batch_size为一组循环
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sample, num_workers=2)   return train_loader

val_loader = torch.utils.data.DataLoader(train_set, batch_size=64, sampler=validation_sample, num_workers=2)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, sampler=test_sample, num_workers=2)
```
2.4 迭代训练并验证

```python
def trainNet(net, batchsize, n_epochs, learning_rate):
    print("HYPERPARAMETERS：")  
    print("batch-size=", batchsize)
    print("n_epochs=", n_epochs)
    print("learning_rate=", learning_rate)

    print("batchsize:", batchsize)
    train_loader = get_train_loader(batchsize)
    n_batches = len(train_loader)  # n_batches * batchsize = 20000（样本数目）
    print("n_batches", n_batches)
    loss, optimizer = createlossandoptimizer(net, learning_rate)
 
    training_start_time = time.time() 
    print("training start:")
    for epoch in range(n_epochs):
        running_loss = 0.0
        print_every = n_batches 
        print("print_every:", print_every)  
        start_time = time.time()
        total_train_loss = 0
 
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            # print(inputs, labels)
            inputs, labels = Variable(inputs), Variable(labels)
# 将所有的梯度置零，原因是防止每次backward的时候梯度会累加
            optimizer.zero_grad() 
            # forward
            outputs = net(inputs)
            # loss
            loss_size = loss(outputs, labels)
            #backward
            loss_size.backward()
            # update weights
            optimizer.step()
            print(loss_size)
            running_loss += loss_size.data[0]
            print("running_loss:", running_loss)
            total_train_loss += loss_size.data[0]
            print("total_train_loss:", total_train_loss)
            # 在一个epoch里。每十组batchsize大小的数据输出一次结果，即以batch_size大小的数据为一组，到第10组，20组，30组...的时候输出
            if (i + 1) % 10 == 0:
                print("epoch{}, {:d} \t traing_loss:{:.2f} took:{:.2f}s".format(epoch + 1, int(100 * (i + 1) / n_batches),
                                                                                running_loss / 10, time.time()-
                                                                                start_time))
                running_loss = 0.0
                start_time = time.time()
 
        total_val_loss = 0
        
        for inputs, labels in val_loader:
            # Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)
 
            # Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels)
            # print("-----", val_loss_size)
            total_val_loss += val_loss_size.data[0]
# 验证集的平均损失          
        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))  

# 所有的Epoch结束，也就是训练结束，计算花费的时间
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))  

```

#### 模型保存、加载

```python
torch.save(model_object.state_dict(), 'model.pt')
model.load_state_dict(torch.load(' model.pt'))
```

## 总结

本节以常用数据集Cifar10为例，用PyTorch训练了一个简单的图像分类器。介绍训练分类器的小目标，结合小目标给出具体步骤，并给出相关知识点及代码。

贡献者：

--- By: 伊雪

--- By: 阿水

    微信公众号：Coggle数据科学

关于Datawhale：

    Datawhale是一个专注于数据科学与AI领域的开源组织，汇集了众多领域院校和知名企业的优秀学习者，聚合了一群有开源精神和探索精神的团队成员。Datawhale以“for the learner，和学习者一起成长”为愿景，鼓励真实地展现自我、开放包容、互信互助、敢于试错和勇于担当。同时Datawhale 用开源的理念去探索开源内容、开源学习和开源方案，赋能人才培养，助力人才成长，建立起人与人，人与知识，人与企业和人与未来的联结。
