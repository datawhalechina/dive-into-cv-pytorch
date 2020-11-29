# Pytorch自动求梯度原理介绍

在深度学习中，我们经常需要对函数求梯度（gradient）。PyTorch提供的[autograd](https://pytorch.org/docs/stable/autograd.html)包能够根据输入和前向传播过程自动构建计算图，并执行反向传播。本节将介绍如何使用autograd包来进行自动求梯度的有关操作。
## 1 基本概念介绍
### 1.1 Variable和Tensor

<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter01/1.3_automatic_gradient/variable.png">

Variable是 torch.autograd中的数据类型，主要用于封装 Tensor，进行自动求导。    
>data : 被包装的Tensor  
grad : data的梯度  
grad\_fn : 创建 Tensor的 Function，是自动求导的关键  
requires_grad：指示是否需要梯度  
is_leaf : 指示是否是叶子结点 

<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter01/1.3_automatic_gradient/tensor.png">

Pytorch 0.4.0版开始，Variable并入Tensor。  
>dtype：张量的数据类型，如torch.FloatTensor，torch.cuda.FloatTensor  
shape：张量的形状，如(64，3，224，224)  
device：张量所在设备，GPU/CPU

`Tensor`是PyTorch实现多维数组计算和自动微分的关键数据结构。一方面，它类似于numpy的ndarray，用户可以对`Tensor`进行各种数学运算；另一方面，当设置`.requires_grad = True`之后，在其上进行的各种操作就会被记录下来，它将开始追踪在其上的所有操作，从而利用链式法则进行梯度传播。完成计算后，可以调用`.backward()`来完成所有梯度计算。此`Tensor`的梯度将累积到`.grad`属性中。

如果不想要被继续追踪，可以调用`.detach()`将其从追踪记录中分离出来，可以防止将来的计算被追踪，这样梯度就传不过去了。此外，还可以用`with torch.no_grad()`将不想被追踪的操作代码块包裹起来，这种方法在评估模型的时候很常用，因为在评估模型时，我们并不需要计算可训练参数（`requires_grad=True`）的梯度。

### 1.2 Function类
`Function`是另外一个很重要的类。`Tensor`和`Function`互相结合就可以构建一个记录有整个计算过程的有向无环图(Directed Acyclic Graph，DAG)。每个`Tensor`都有一个`.grad_fn`属性，该属性即创建该`Tensor`的`Function`，就是说该`Tensor`是不是通过某些运算得到的，若是，则`grad_fn`返回一个与这些运算相关的对象，否则是None。

我们已经知道PyTorch使用有向无环图DAG记录计算的全过程，那么DAG是怎样建立的呢？DAG的节点是`Function`对象，边表示数据依赖，从输出指向输入。
每当对`Tensor`施加一个运算的时候，就会产生一个`Function`对象，它产生运算的结果，记录运算的发生，并且记录运算的输入。`Tensor`使用`.grad_fn`属性记录这个计算图的入口。反向传播过程中，`autograd`引擎会按照逆序，通过`Function`的`backward`依次计算梯度。

<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter01/1.3_automatic_gradient/computational_graph.gif">

### 1.3 代码示例
创建一个`Tensor`并设置`requires_grad=True`:
``` python
x = torch.ones(2, 2, requires_grad=True)
print(x)
print(x.grad_fn)
```
输出：
```
tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
None
```
再做一下运算操作：
``` python
y = x + 2
print(y)
print(y.grad_fn)
```
输出：
```
tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward>)
<AddBackward object at 0x1100477b8>
```
注意x是直接创建的，所以它没有`grad_fn`, 而y是通过一个加法操作创建的，所以它有一个为`<AddBackward>`的`grad_fn`。

像x这种直接创建的称为叶子节点，叶子节点对应的`grad_fn`是`None`。
``` python
print(x.is_leaf, y.is_leaf) # True False
```

再来点复杂度运算操作：
``` python
z = y * y * 3
out = z.mean()
print(z, out)
```
输出：
```
tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward>) tensor(27., grad_fn=<MeanBackward1>)
```

通过`.requires_grad_()`来用in-place的方式改变`requires_grad`属性：
``` python
a = torch.randn(2, 2) # 缺失情况下默认 requires_grad = False
a = ((a * 3) / (a - 1))
print(a.requires_grad) # False
a.requires_grad_(True)
print(a.requires_grad) # True
b = (a * a).sum()
print(b.grad_fn)
```
输出：
```
False
True
<SumBackward0 object at 0x118f50cc0>
```

## 2 autograd 自动求梯度
深度学习模型的训练就是不断更新权值，权值的更新需要求解梯度，梯度在模型训练中是至关重要的。Pytorch提供自动求导系统，我们不需要手动计算梯度，只需要搭建好前向传播的计算图，然后根据Pytorch中的`autograd`方法就可以得到所有张量的梯度。
PyTorch中，所有神经网络的核心是`autograd`包。`autograd`包为张量上的所有操作提供了自动求导机制。它是一个在运行时定义（define-by-run）的框架，这意味着反向传播是根据代码如何运行来决定的，并且每次迭代可以是不同的。

### 2.1 torch.autograd.backward 

``` python
torch.autograd.backward(tensors,
                        grad_tensors=None,
                        retain_grad=None,
                        create_graph=False)
```


>功能：自动求取梯度  
tensors: 用于求导的张量，如loss  
retain_graph : 保存计算图；由于pytorch采用动态图机制，在每一次反向传播结束之后，计算图都会释放掉。如果想继续使用计算图，就需要设置参数retain_graph为True  
create_graph : 创建导数计算图，用于高阶求导，例如二阶导数、三阶导数等  
grad_tensors：多梯度权重；当有多个loss需要去计算梯度的时候，就要设计各个loss之间的权重比例  


### 2.2 torch.autograd.grad 

``` python
torch.autograd.grad(outputs,
                    inputs,
                    grad_outputs=None,
                    retain_graph=None,
                    create_graph=False)
```
>功能：计算并返回outputs对inputs的梯度  
outputs：用于求导的张量，如loss  
inputs：需要梯度的张量，如w   
create_graph：创建导数计算图，用于高阶求导   
retain_graph：保存计算图  
grad_outputs：多梯度权重   


### 2.3 链式法则

数学上，如果有一个函数值和自变量都为向量的函数 $\vec{y}=f(\vec{x})$, 那么 $\vec{y}$ 关于 $\vec{x}$ 的梯度就是一个雅可比矩阵（Jacobian matrix）:
$$
J=\left(\begin{array}{ccc}
   \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
   \vdots & \ddots & \vdots\\
   \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
   \end{array}\right)
$$
而``torch.autograd``这个包就是用来计算一些雅克比矩阵的乘积的。例如，如果 $v$ 是一个标量函数的 $l=g\left(\vec{y}\right)$ 的梯度：
$$
v=\left(\begin{array}{ccc}\frac{\partial l}{\partial y_{1}} & \cdots & \frac{\partial l}{\partial y_{m}}\end{array}\right)
$$
那么根据链式法则我们有 $l$ 关于 $\vec{x}$ 的雅克比矩阵就为:
$$
v J=\left(\begin{array}{ccc}\frac{\partial l}{\partial y_{1}} & \cdots & \frac{\partial l}{\partial y_{m}}\end{array}\right) \left(\begin{array}{ccc}
   \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
   \vdots & \ddots & \vdots\\
   \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
   \end{array}\right)=\left(\begin{array}{ccc}\frac{\partial l}{\partial x_{1}} & \cdots & \frac{\partial l}{\partial x_{n}}\end{array}\right)
$$

注意：grad在反向传播过程中是累加的(accumulated)，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以一般在反向传播之前需把梯度清零。

### 2.4 代码示例
``` python
import torch
torch.manual_seed(10)  #用于设置随机数

w = torch.tensor([1.], requires_grad=True)    #创建叶子张量，并设定requires_grad为True，因为需要计算梯度；
x = torch.tensor([2.], requires_grad=True)    #创建叶子张量，并设定requires_grad为True，因为需要计算梯度；

a = torch.add(w, x)    #执行运算并搭建动态计算图
b = torch.add(w, 1)
y = torch.mul(a, b)

y.backward(retain_graph=True)   
print(w.grad)   #输出为tensor([5.])
``` 
从代码中可以发现对y求导使用的是y.backward()方法，也就是张量中的类方法。我们上面介绍的是torch.autograd中的backward()。这两个方法之间有什么联系呢？
通过pycharm中的断点调试，可以发现y.backward()是Tensor.py中的一个类方法的函数。这个函数只有一行代码，就是调用torch.autograd.backward()。
``` python
def backward(self, gradient=None, retain_graph=None, create_graph=False):
    torch.autograd.backward(self, gradient, retain_graph, create_graph)
```
从代码调试中可以知道张量中的backward()方法实际直接调用了torch.autograd中的backward()。
backward()中有一个retain_grad参数，它是用来保存计算图的，如果还想执行一次反向传播 ，必须将retain_grad参数设置为retain_grad=True，否则代码会报错。因为如果没有retain_grad=True，每进行一次backward之后，计算图都会被清空，没法再进行一次backward()操作。

### 2.5 关于y.backward()

**为什么在`y.backward()`时，如果`y`是标量，则不需要为`backward()`传入任何参数；否则，需要传入一个与`y`同形的`Tensor`?**

>简单来说就是为了避免向量（甚至更高维张量）对张量求导，而转换成标量对张量求导。举个例子，假设形状为 `m x n` 的矩阵 X 经过运算得到了 `p x q` 的矩阵 Y，Y 又经过运算得到了 `s x t` 的矩阵 Z。那么按照前面讲的规则，dZ/dY 应该是一个 `s x t x p x q` 四维张量，dY/dX 是一个 `p x q x m x n`的四维张量。问题来了，怎样反向传播？怎样将两个四维张量相乘？？？这要怎么乘？？？就算能解决两个四维张量怎么乘的问题，四维和三维的张量又怎么乘？导数的导数又怎么求，这一连串的问题，感觉要疯掉…… 

>为了避免这个问题，我们**不允许张量对张量求导，只允许标量对张量求导，求导结果是和自变量同形的张量**。所以必要时我们要把张量通过将所有张量的元素加权求和的方式转换为标量，举个例子，假设`y`由自变量`x`计算而来，`w`是和`y`同形的张量，则`y.backward(w)`的含义是：先计算`l = torch.sum(y * w)`，则`l`是个标量，然后求`l`对自变量`x`的导数。

来看一些实际例子。
``` python
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = 2 * x
z = y.view(2, 2)
print(z)
```
输出：
```
tensor([[2., 4.],
        [6., 8.]], grad_fn=<ViewBackward>)
```
现在 `z` 不是一个标量，所以在调用`backward`时需要传入一个和`z`同形的权重向量进行加权求和得到一个标量。
``` python
v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
z.backward(v)
print(x.grad)
```
输出：
```
tensor([2.0000, 0.2000, 0.0200, 0.0020])
```
注意，`x.grad`是和`x`同形的张量。

再来看看中断梯度追踪的例子：
``` python
x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2 
with torch.no_grad():
    y2 = x ** 3
y3 = y1 + y2
    
print(x.requires_grad)
print(y1, y1.requires_grad) # True
print(y2, y2.requires_grad) # False
print(y3, y3.requires_grad) # True
```
输出：
```
True
tensor(1., grad_fn=<PowBackward0>) True
tensor(1.) False
tensor(2., grad_fn=<ThAddBackward>) True
```
可以看到，上面的`y2`是没有`grad_fn`而且`y2.requires_grad=False`的，而`y3`是有`grad_fn`的。如果我们将`y3`对`x`求梯度的话会是多少呢？
``` python
y3.backward()
print(x.grad)
```
输出：
```
tensor(2.)
```
为什么是2呢？$ y_3 = y_1 + y_2 = x^2 + x^3$，当 $x=1$ 时 $\frac {dy_3} {dx}$ 不应该是5吗？事实上，由于 $y_2$ 的定义是被`torch.no_grad():`包裹的，所以与 $y_2$ 有关的梯度是不会回传的，只有与 $y_1$ 有关的梯度才会回传，即 $x^2$ 对 $x$ 的梯度。

上面提到，`y2.requires_grad=False`，所以不能调用 `y2.backward()`，会报错：
```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

此外，如果我们想要修改`tensor`的数值，但是又不希望被`autograd`记录（即不会影响反向传播），那么我么可以对`tensor.data`进行操作。
``` python
x = torch.ones(1,requires_grad=True)

print(x.data) # 还是一个tensor
print(x.data.requires_grad) # 但是已经是独立于计算图之外

y = 2 * x
x.data *= 100 # 只改变了值，不会记录在计算图，所以不会影响梯度传播

y.backward()
print(x) # 更改data的值也会影响tensor的值
print(x.grad)
```
输出：
```
tensor([1.])
False
tensor([100.], requires_grad=True)
tensor([2.])
```


### 注意事项

* 梯度不自动清零，如果不清零梯度会累加，所以需要在每次梯度后人为清零。
* 依赖于叶子结点的结点，requires_grad默认为True。
* 叶子结点不可执行in-place，因为其他节点在计算梯度时需要用到叶子节点，所以叶子地址中的值不得改变否则会是其他节点求梯度时出错。所以叶子节点不能进行原位计算。
* 注意在y.backward()时，如果y是标量量，则不需要为backward()传⼊入任何参数；否则，需要传⼊一个与y同形的Tensor。

-----------

**贡献者**

作者: [星尘](https://blog.csdn.net/OuDiShenmiss)

> 本文内容参考 [Dive-into-DL-PyTorch](https://github.com/ShusenTang/Dive-into-DL-PyTorch/edit/master/docs/chapter02_prerequisite/2.3_autograd.md)

