import torch

#--------------
# tensors
#--------------

x = torch.empty(5, 3)
print(x)
print(type(x))

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3)
print(x)
print(x.dtype)
x = torch.zeros(5, 3, dtype=torch.long)
print(x.dtype)

x = torch.tensor([5.5, 3])
print(x)

x = torch.tensor([5.5, 3], dtype=torch.double)
print(x.dtype)
x = x.new_ones(5, 3)      # new_* methods take in sizes
print(x)
x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size

x_size = x.size()
print(x_size)
row, col = x_size
print(row, col)


#--------------
# Operations
#--------------

y = torch.rand(5, 3)
z1 = x + y
print(z1)

z2 = torch.add(x, y)
print(z2)

z3 = torch.empty(5, 3)
torch.add(x, y, out=z3)
print(z3)

y.add_(x)
print(y)

print(x[:, 1])

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)   # 使用-1时pytorch将会自动根据其他维度进行推导
print(x.size(), y.size(), z.size())

x = torch.randn(1)
print(x)
print(x.size())
print(x.item())
print(type(x.item()))


#--------------
# NumPy Bridge
#--------------

a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)


#--------------
# CUDA Tensors
#--------------
print(torch.cuda.is_available())

# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
