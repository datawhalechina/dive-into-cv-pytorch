import torch
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

# plot src data
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo')
plt.show()

# define the linear model 
# y = w * x + b
w = torch.tensor([-1.], requires_grad=True)
b = torch.tensor([0.], requires_grad=True)
def linear_model(x):
    return x * w + b

# plot estimate result before train
y_ = linear_model(x_train)
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='estimated')
plt.legend()
plt.show()

# define the loss function (最小二乘)
def get_loss(y_, y): 
	return torch.mean((y_ - y_train) ** 2)

# train 10 iteration
lr = 1e-2
for e in range(10):
    y_ = linear_model(x_train) 

    # compute loss
    loss = get_loss(y_, y_train) 
    loss.backward() 

    # 手动更新参数
    w.data = w.data - lr * w.grad.data
    b.data = b.data - lr * b.grad.data
    print('epoch: {}, loss: {}'.format(e, loss))

    # reset the grad to zero
    w.grad.zero_()
    b.grad.zero_()

# plot estimate result after train
y_ = linear_model(x_train)
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='estimated')
plt.legend()
plt.show()
