# Pytorch 线性回归实战

通过前面的学习我们已经熟知了pytorch的基本用法和自动求梯度的基本原理，这节课我们将结合前面所学的知识做一个简单的demo，我们将使用pytorch来构建一个简单的线性回归实战案例并且将训练模型的过程进行一次总结

## 1.线性回归模型

1. 什么是线性回归模型？我们学过因变量关于自变量的线性关系，即：在数据呈现随着自变量增长，因变量也呈现相应增长。如果数据落在直线 $y =  ax+b$ 上，则称自变量 $x$ 与因变量 $y$ 成线性关系。那么对于无法落在一条直线上却明显的呈线性条带状分布的数据，我们认为这是线性关系的数据参杂了一定噪声之后呈现出的情形。因此我们想尽可能地用线性函数拟合分布曲线，得到更好的预测结果，之后可以通过 $x$ 来算出 $y$ 的预测值。

2. 这里简单介绍一下一元线性回归模型: 利用线性函数(一阶或更低阶多项式)对自变量向量 $ x = (x_1,x_2,...,x_k)^T \{k \in \Z\} $ 和因变量 $ y $ 之间的关系进行拟合的模型。通常使用的表达式为: $\hat y = h(x) = a + b_1x_1 + b_2x_2 + \cdots + b_nx_n$,  其中 $\hat y$ 是我们的预测值，我们将目标 $y$ 与预测值 $\hat y$ 进行对比之后通过最小二乘拟合法求目标函数 $h(x)$ ，在此我们采用前面所学的梯度下降的方式进行优化。

## 2.梯度下降算法简略介绍

梯度下降算法是一种通过迭代找到目标函数的极小值，或者收敛到极小值的方法，适合用于无法对函数全局有掌握时希望找到函数极小值的方法。如果已知函数全局为凸函数，则可以很好的收敛到全局最小值，而这里的最小二乘拟合法正是寻找凸函数最小值的情形。

如果读者掌握基本的数学分析功底，则可以轻松推出函数的梯度与函数的等势面垂直，因此梯度方向是函数值变化最快的方向，对于可微函数，我们通过反复迭代，每一步向当前点梯度方向下降一定距离的方法，便能得到极小值。

<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter01/1.4_linear_regression_pytorch/grad_decent.png">

## 3.反向传播算法简略介绍

反向传播算法是对多层模型使用梯度下降算法时的方法，通常我们会在神经网络的训练中见到他，事实上如果将网络的每一层视为一个感知机，那么这便是一个多层模型。那么对多层模型参数不容易轻易的表示为显式形式时，我们该怎么使用梯度下降算法呢？事实上有多种理解，例如将误差反向传播直隐藏层以计算梯度等，这里我们使用朴素的数学理解方法：

1. 对每一层的参数进行变化时，我们通过函数的复合可以显式的将损失函数表达为所求参数和该层的输入的复合函数，即将该层的输入视为模型的输入，将该层的输出视为模型输出，该层之后的所有层与损失函数合起来认为是新损失函数。只需对该层进行正常的梯度下降即可。

2. 在梯度下降过程中我们遇到了复合函数求导的问题，这时候就要使用链式法则，也就是反向传播算法里唯一复杂的地方。

   $$
   \frac{\partial L}{∂ω^n}=\frac{\partial L}{∂x^{n+1}}\cdot\frac{∂x^{n+1}}{∂ω^n}=\frac{\partial L}{∂x^{n+2}}\cdot\frac{\partial x^{n+2}}{∂x^{n+1}}\cdot\frac{∂x^{n+1}}{∂ω^n}=\cdots
   $$

3. 掌握了链式法则之后，反向传播便回归了简单的梯度下降算法问题了。

## 4.基本流程

利用 pytorch 训练模型的过程大致分为一下五个部分: 

1. 获取输入数据 如这里我们手动定义一些数据

```
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
[9.779], [6.182], [7.59], [2.167], [7.042], 
[10.791], [5.313], [7.997], [3.1]], dtype=np.float32) 
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
[3.366], [2.596], [2.53], [1.221], [2.827], 
[3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
```

2. 定义参数和模型 构建一个未初始化的5*3的空矩阵（张量）

```python
#我们定义好参数w如下将w定义为一个随机的正态分布
W = Variable(torch.randn(1),requires_grad=True)
#b定义成零矩阵:
b = Variable(torch.zeros(1),requires_grad=True)
#定义线性模型:
def linar_model(x):
return w *x +b
```

3. 定义损失函数 其中y_预测值,y真实值

```python
def get_loss(y_,y):
	return torch.mean((y_ - y )**2)
```

4. 反向传播，得到梯度

```python
loss.backward()(反向求梯度)
```

5. 梯度下降进行优化更新模型参数

```python
#进行10次数据更新
for e in range(10):
	Y_ = linear_model(x_train)
	Loss = get_loss(y_,y_train)
    #梯度归零
	w.grad.zero_()
    b.grad.zero_()
    #反向传播计算梯度
    Loss.backward()
    #梯度下降
    w.data = w.data -le-2*w.grad.data
    b.data = b.data -le-2*bgrad.data
    Print(‘epoch:{},loss:{}’,format(e,loss.data[0])
```

## 5.项目实战 下面我们创建一个随机初始化的矩阵

```python
首先我们导入基本项目库
import torch 
import numpy as np 
from torch.autograd import Variable
#画图象
import matplotlib.pyplot as plt
```

1.导入相关数据

```python
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
[9.779], [6.182], [7.59], [2.167], [7.042], 
[10.791], [5.313], [7.997], [3.1]], dtype=np.float32) 
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
[3.366], [2.596], [2.53], [1.221], [2.827], 
[3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
#转换为 Tensor
x_train = torch.from_numpy(x_train) 
y_train = torch.from_numpy(y_train)
```

2.定义 W 和 b

```python
w = Variable(torch.randn(1), requires_grad=True)
b = Variable(torch.zeros(1), requires_grad=True)
```

3.构建模型

```python
x_train = Variable(x_train) 
y_train = Variable(y_train)
y_ = linear_model(x_train)
```

4.构建损失函数

```python
def get_loss(y_, y): 
	return torch.mean((y_ - y_train) ** 2)
loss = get_loss(y_, y_train)
```

5.反向传播求梯度

```python
loss.backward()
```

6.利用梯度下降优化

```python
for e in range(10):
	y_ = linear_model(x_train) 
	loss = get_loss(y_, y_train) 
	w.grad.zero_()
	b.grad.zero_()
	loss.backward() 
	w.data = w.data - 1e-2 * w.grad.data
	b.data = b.data - 1e-2 * b.grad.data
	print('epoch: {}, loss: {}'.format(e, loss.data[0]))
```

7.查看结果

```python
epoch:	0,	loss:	3.1357719898223877
epoch:	1,	loss:	0.3550889194011688
epoch:	2,	loss:	0.30295443534851074
epoch:	3,	loss:	0.30131956934928894
epoch:	4,	loss:	0.3006229102611542
epoch:	5,	loss:	0.29994693398475647
epoch:	6,	loss:	0.299274742603302
epoch:	7,	loss:	0.2986060082912445
epoch:	8,	loss:	0.2979407012462616
epoch:	9,	loss:	0.29727882146835327
```

**关于 Datawhale**：

> Datawhale 是一个专注于数据科学与AI领域的开源组织，汇集了众多领域院校和知名企业的优秀学习者，聚合了一群有开源精神和探索精神的团队成员。Datawhale 以“ for the learner，和学习者一起成长 ”为愿景，鼓励真实地展现自我、开放包容、互信互助、敢于试错和勇于担当。同时 Datawhale   用开源的理念去探索开源内容、开源学习和开源方案，赋能人才培养，助力人才成长，建立起人与人，人与知识，人与企业和人与未来的联结。
