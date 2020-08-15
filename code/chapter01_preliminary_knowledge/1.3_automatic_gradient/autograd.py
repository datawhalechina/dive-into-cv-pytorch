import torch

#创建一个`Tensor`并设置`requires_grad=True`
x = torch.ones(2, 2, requires_grad=True)
print(x)
print(x.grad_fn)

# 再做一下运算操作：
y = x + 2
print(y)
print(y.grad_fn)

#像x这种直接创建的称为叶子节点，叶子节点对应的`grad_fn`是`None`。
print(x.is_leaf, y.is_leaf) # True False

#再来点复杂度运算操作：
z = y * y * 3
out = z.mean()
print(z, out)

#通过`.requires_grad_()`来用in-place的方式改变`requires_grad`属性：
a = torch.randn(2, 2) # 缺失情况下默认 requires_grad = False
a = ((a * 3) / (a - 1))
print(a.requires_grad) # False
a.requires_grad_(True)
print(a.requires_grad) # True
b = (a * a).sum()
print(b.grad_fn)


torch.manual_seed(10)  #用于设置随机数
w = torch.tensor([1.], requires_grad=True)    #创建叶子张量，并设定requires_grad为True，因为需要计算梯度；
x = torch.tensor([2.], requires_grad=True)    #创建叶子张量，并设定requires_grad为True，因为需要计算梯度；
a = torch.add(w, x)    #执行运算并搭建动态计算图
b = torch.add(w, 1)
y = torch.mul(a, b)
y.backward(retain_graph=True)
print(w.grad)   #输出为tensor([5.])


x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = 2 * x
z = y.view(2, 2)
print(z)

#现在 `z` 不是一个标量，所以在调用`backward`时需要传入一个和`z`同形的权重向量进行加权求和得到一个标量。
v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
z.backward(v)
print(x.grad)

#再来看看中断梯度追踪的例子：
x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2
with torch.no_grad():
    y2 = x ** 3
y3 = y1 + y2
print(x.requires_grad)
print(y1, y1.requires_grad)  # True
print(y2, y2.requires_grad)  # False
print(y3, y3.requires_grad)  # True

#可以看到，上面的`y2`是没有`grad_fn`而且`y2.requires_grad=False`的，而`y3`是有`grad_fn`的。如果我们将`y3`对`x`求梯度的话会是多少呢？
y3.backward()
print(x.grad)

#此外，如果我们想要修改`tensor`的数值，但是又不希望被`autograd`记录（即不会影响反向传播），那么我么可以对`tensor.data`进行操作。
x = torch.ones(1,requires_grad=True)
print(x.data) # 还是一个tensor
print(x.data.requires_grad) # 但是已经是独立于计算图之外
y = 2 * x
x.data *= 100 # 只改变了值，不会记录在计算图，所以不会影响梯度传播
y.backward()
print(x) # 更改data的值也会影响tensor的值
print(x.grad)
