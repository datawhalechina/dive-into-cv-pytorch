# BaseLine plus

今天又做了一些简单改动，将分数提到了0.9526:rocket:

调参的思路如下：

### 调整输入的归一化系数

对输入进行归一化是比较通用的做法

归一化时需要统一进行 x = (x - mean) / std 的操作

之前baseline的这两个系数我是随便给点，这里调整为整个数据集的mean和std

我们首先取消`load_data_fashion_mnist`函数中的归一化操作，然后用如下代码计算整个数据集的归一化系数：

```python
# 求整个数据集的均值
temp_sum = 0
cnt = 0
for X, y in train_iter:
    if y.shape[0] != batch_size:
        break   # 最后一个batch不足batch_size,这里就忽略了
    channel_mean = torch.mean(X, dim=(0,2,3))  # 按channel求均值(不过这里只有1个channel)
    cnt += 1   # cnt记录的是batch的个数，不是图像
    temp_sum += channel_mean[0].item()
dataset_global_mean = temp_sum / cnt
print('整个数据集的像素均值:{}'.format(dataset_global_mean))
# 求整个数据集的标准差
cnt = 0
temp_sum = 0
for X, y in train_iter:
    if y.shape[0] != batch_size:
        break   # 最后一个batch不足batch_size,这里就忽略了
    residual = (X - dataset_global_mean) ** 2
    channel_var_mean = torch.mean(residual, dim=(0,2,3))
    cnt += 1   # cnt记录的是batch的个数，不是图像
    temp_sum += math.sqrt(channel_var_mean[0].item())
dataset_global_std = temp_sum / cnt
print('整个数据集的像素标准差:{}'.format(dataset_global_std))
```

运行结果：

```
计算数据集均值标准差
整个数据集的像素均值:0.2860483762389695
整个数据集的像素标准差:0.3529184201347597
```

### 进行数据增强

之前的baseline没用数据增强，这里就简单加上两种最简单也最有效的数据增强方法

RandomCrop 和 RandomHorizontalFlip

修改`load_data_fashion_mnist`函数如下：

```python
# 定义加载数据集的函数
def load_data_fashion_mnist(batch_size, root='../../dataset', use_normalize=False, mean=None, std=None):
    """Download the fashion mnist dataset and then load into memory."""

    if use_normalize:
        normalize = transforms.Normalize(mean=[mean], std=[std])
        train_augs = transforms.Compose([transforms.RandomCrop(28, padding=2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize])
        test_augs = transforms.Compose([transforms.ToTensor(), normalize])
    else:
        train_augs = transforms.Compose([transforms.ToTensor()])
        test_augs = transforms.Compose([transforms.ToTensor()])

    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=train_augs)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=test_augs)
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter
```

### 修改优化方法

adam 和 sgd 是两种最常用的优化方法

adam的优势在于收敛快，而这里反正数据集小网络小，我们可以使用sgd来充分的训一次试一下。

```python
#optimizer = optim.Adam(net.parameters(), lr=lr)
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) 
```


### 训练

完成以上改动，开始训练，可以得到一个0.94分以上的baseline：

```
训练...
epoch 1, loss 0.5300, train acc 0.8015, test acc 0.8507, time 20.9 sec
find best! save at model/best.pth
epoch 2, loss 0.3289, train acc 0.8794, test acc 0.8708, time 20.3 sec
find best! save at model/best.pth
...
...
epoch 49, loss 0.1241, train acc 0.9554, test acc 0.9375, time 19.7 sec
epoch 50, loss 0.1242, train acc 0.9554, test acc 0.9403, time 19.3 sec
find best! save at model/best.pth
加载最优模型
inference测试集
生成提交结果文件
```

仅供刚入门的小伙伴参考，欢迎把进一步提分的技巧和喜悦分享给我~


## 2020/2/26更新 0.9526:rocket:

仅仅在之前的基础上，添加学习率阶段性下降的策略，即可将分数进一步提升到0.9526

在train_model函数中加入如下逻辑即可

```python
def train_model(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs, lr, lr_period, lr_decay):
    ...
    ... 
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        if epoch > 0 and epoch % lr_period == 0:  # 每lr_period个epoch，学习率衰减一次
            lr = lr * lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            ...
            ...
```
