# Pytorch基础使用介绍

Pytorch是什么？

Pytorch是一个基于python的科学计算包，主要面向两部分受众：

- 一个为了充分发挥GPU能力而设计的Numpy的GPU版本替代方案

- 一个提供更大灵活性和速度的深度学习研究平台

本节将会介绍Pytorch的一些基本使用和操作，内容高度依赖Pytorch官方[What is Pytorch?](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py)教程。

让我们开始吧～

## 1.Tensors

Tensors(张量)的概念可以类比Numpy中的ndarrays，本质上就是一个多维数组，是任何运算和操作间数据流动的最基础形式。

首先让我们加载torch库

```python
import torch
```

构建一个未初始化的5\*3的空矩阵（张量）

```python
x = torch.empty(5, 3)
print(type(x))
print(x)
```

输出：

```
<class 'torch.Tensor'>
tensor([[7.7050e+31, 6.7415e+22, 1.2690e+31],
        [6.1186e-04, 4.6165e+24, 4.3701e+12],
        [7.5338e+28, 7.1774e+22, 3.7386e-14],
        [6.6532e-33, 1.8337e+31, 1.3556e-19],
        [1.8370e+25, 2.0616e-19, 4.7429e+30]])
```

注意，对于未初始化的张量，它的取值是不固定的，取决于它创建时分配的那块内存的取值。

下面我们创建一个随机初始化的矩阵

```python
x = torch.rand(5, 3)
print(x)
```

输出：

```
tensor([[0.6544, 0.1733, 0.2569],
        [0.0680, 0.5781, 0.2667],
        [0.5051, 0.5366, 0.2776],
        [0.4903, 0.9934, 0.1181],
        [0.4497, 0.6201, 0.1952]])
```

构建一个使用0填充的tensor，并尝试dtype属性的设置，观察区别

```
x = torch.zeros(5, 3)
print(x)
print(x.dtype)
x = torch.zeros(5, 3, dtype=torch.long)                                                          
print(x.dtype)
```

输出：

```
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
torch.float32
torch.int64
```

和Numpy类似，除了常见的0/1取值的初始化，我们还可以直接用现有数据进行张量的初始化

```python
x = torch.tensor([5.5, 3])
print(x)
```

输出：

```
tensor([5.5000, 3.0000])
```

我们还可以基于已有的tensor来创建新的tensor，通常是为了复用已有tensor的一些属性，包括shape和dtype。观察下面的示例：

```python
x = torch.tensor([5.5, 3], dtype=torch.double)
print(x.dtype)
x = x.new_ones(5, 3)      # new_* methods take in sizes
print(x)
x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size
```

输出：

```
torch.float64
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
tensor([[-0.4643, -0.1942, -0.4497],
        [ 0.8090,  1.6753, -0.0678],
        [-1.1654, -0.3817,  0.3495],
        [ 2.3517,  1.7688,  0.0251],
        [-0.6314, -1.2776,  0.8398]])
```

可以看到，`new_ones`函数复用了x的`dtype`属性，`randn_like`函数复用了x的`shape`同时通过手动指定数据类型覆盖了原有的`dtype`属性

如果获取tensor的size？很简单

```python
x_size = x.size()
print(x_size)
row, col = x_size
print(row, col)
```

输出：

```
torch.Size([5, 3])
5 3
```

`torch.Size`本质上是一个`tuple`，通过上面的例子也可以看出，它支持元组的操作。

## 2.Operations

Operations(操作)设计的语法很多，但很多都是相通的，下面我们来看看加法操作作为示例。

加法：语法1

```python
y = torch.rand(5, 3)
z1 = x + y
print(z1)
```

输出：

```
tensor([[ 1.7110, -0.0545,  0.3557],
        [ 0.1074,  2.3876,  1.2455],
        [-0.1204,  1.1829,  0.6250],
        [ 0.0873,  1.3114,  1.0403],
        [-0.6172,  1.2261, -1.0718]])
```

加法：语法2

```python
z2 = torch.add(x, y)
print(z2)
```

输出：

```
tensor([[ 1.7110, -0.0545,  0.3557],
        [ 0.1074,  2.3876,  1.2455],
        [-0.1204,  1.1829,  0.6250],
        [ 0.0873,  1.3114,  1.0403],
        [-0.6172,  1.2261, -1.0718]])
```

加法：语法3，通过参数的形式进行输出tensor的保存

```python
z3 = torch.empty(5, 3)
torch.add(x, y, out=z3)
print(z3)
```

输出：

```
tensor([[ 1.7110, -0.0545,  0.3557],
        [ 0.1074,  2.3876,  1.2455],
        [-0.1204,  1.1829,  0.6250],
        [ 0.0873,  1.3114,  1.0403],
        [-0.6172,  1.2261, -1.0718]])
```

加法：语法4，通过in-place操作直接将计算结果覆盖到y上

```python
y.add_(x)
print(y)  
```

输出：

```
tensor([[ 1.7110, -0.0545,  0.3557],
        [ 0.1074,  2.3876,  1.2455],
        [-0.1204,  1.1829,  0.6250],
        [ 0.0873,  1.3114,  1.0403],
        [-0.6172,  1.2261, -1.0718]])
```

在Pytorch中，我们约定凡是会覆盖函数调用主体的`in-place`操作，都以后缀`_`结束，例如：`x.copy_(y)`，`x.t_()`，都会改变`x`的取值。

下面我们来看看其他的一些基础操作。

在Pytorch中，我们可以使用标准的`Numpy-like`的索引操作，例如：

```python
print(x[:, 1])
```

输出：

```
tensor([-0.0211, -1.4157,  0.1453, -1.3401, -0.8556])
```

Resize操作：

如果你想要对tensor进行类似`resize/reshape`的操作，你可以使用`torch.view`

```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)   # 使用-1时pytorch将会自动根据其他维度进行推导
print(x.size(), y.size(), z.size())  
```

输出：

```
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
```

如果有一个tensor，只包含一个元素，你可以使用`.item()`来获得它对应的Python形式的取值

```python
x = torch.randn(1)
print(x)
print(x.size())
print(x.item())
print(type(x.item())) 
```

输出：

```
tensor([0.4448])
torch.Size([1])
0.4447539746761322
<class 'float'>
```

想要学习更多？[这里](https://pytorch.org/docs/stable/torch.html)的官方教程有更多关于tensor操作的介绍，介绍了100多个Tensor运算，包括转置，索引，切片，数学运算，线性代数，随机数等。

## 3.Numpy桥梁

Pytorch中可以很方便的将Torch的Tensor同Numpy的ndarray进行互相转换，相当于在Numpy和Pytorch间建立了一座沟通的桥梁，这将会让我们的想法实现起来变得非常方便。

注：Torch Tensor 和 Numpy ndarray 底层是分享内存空间的，也就是说改变其中之一会同时改变另一个（前提是你是在CPU上使用Torch Tensor）。

将一个Torch Tensor 转换为 Numpy Array

```python
a = torch.ones(5)
print(a)
```

输出：

```
tensor([1., 1., 1., 1., 1.])
```

```python
b = a.numpy()
print(b)
```

输出：

```
[1. 1. 1. 1. 1.]
```

我们来验证下它们的取值是如何互相影响的

```python
a.add_(1)
print(a)
print(b)
```

输出：

```
tensor([2., 2., 2., 2., 2.])
[2. 2. 2. 2. 2.]
```

将一个 Numpy Array 转换为 Torch Tensor

```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b) 
```

输出：

```
[2. 2. 2. 2. 2.]
tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
```

注：所有CPU上的Tensors，除了CharTensor均支持与Numpy的互相转换

## 4.CUDA Tensors

`1.1 深度学习环境配置`一节，我们介绍了如何配置深度学习环境，以及如何安装GPU版本的pytorch，可以通过以下代码进行验证：

```python
print(torch.cuda.is_available())
```

如果输出`True`，代表你安装了GPU版本的pytorch

```
True
```

Tensors可以通过`.to`函数移动到任何我们定义的设备`device`上，观察如下代码：

```python
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!  
```

tensor([0.5906], device='cuda:0')
tensor([0.5906], dtype=torch.float64)


---

--- ***By: 安晟***

>一只普通的算法攻城狮，邮箱[anshengmath@163.com]，[CSDN博客](https://blog.csdn.net/u011583927)，[Github](https://github.com/monkeyDemon)


**关于Datawhale**：

>Datawhale是一个专注于数据科学与AI领域的开源组织，汇集了众多领域院校和知名企业的优秀学习者，聚合了一群有开源精神和探索精神的团队成员。Datawhale以“for the learner，和学习者一起成长”为愿景，鼓励真实地展现自我、开放包容、互信互助、敢于试错和勇于担当。同时Datawhale 用开源的理念去探索开源内容、开源学习和开源方案，赋能人才培养，助力人才成长，建立起人与人，人与知识，人与企业和人与未来的联结。
