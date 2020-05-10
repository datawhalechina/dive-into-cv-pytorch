# 天池计算机视觉入门赛:SVHN数据集实战

## 比赛简介

TODO：比赛介绍见×××

baseline思路：使用CNN进行定长字符分类；

TODO: 数据集下载链接 ×××

注：以下代码默认已将比赛数据的根文件夹命名为`tianchi_SVHN`并放置于`Dive-into-CV-PyTorch/dataset/tianchi_SVHN`下


## 环境安装

本节所介绍的实战代码对环境有特殊依赖，理论上Python2/3，Pytorch1.x版本均可以跑通。此外，由于数据集较小，有无GPU都是可以的，动手开始实战吧~

下面给出 python3.7 + torch1.3.1-gpu版本的环境安装示例

注：假设你已经安装了Anaconda及CUDA10.0,如果你对环境安装不太了解，请阅读第一章的环境安装部分教程。

首先在Anaconda中创建一个专门用于本次天池联系赛的虚拟环境。

`$ conda create -n py37_torch131 python=3.7`

然后激活环境，并安装gpu版本Pytorch

`$ source activate py37_torch131`

`$ conda install pytorch=1.3.1 torchvision cudatoolkit=10.0`

最后通过下面的命令一键完成其他依赖库的安装。

`$ pip install tqdm pandas matplotlib opencv-python jupyter`


## 首先导入必要的库

```python
# -*- coding: utf-8 -*-                                                                          
import os, sys, glob, shutil, json
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import cv2
import numpy as np
from PIL import Image

from tqdm import tqdm, tqdm_notebook

%pylab inline

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
```

## 定义读取数据集

```python
class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl)  + (5 - len(lbl)) * [10]
        return img, torch.from_numpy(np.array(lbl[:5]))

    def __len__(self):
        return len(self.img_path)

```


## 定义读取数据dataloader

```python
train_path = glob.glob('../../../dataset/tianchi_SVHN/train/*.png')
train_path.sort()
train_json = json.load(open('../../../dataset/tianchi_SVHN/train.json'))
train_label = [train_json[x]['label'] for x in train_json]
print(len(train_path), len(train_label))

train_loader = torch.utils.data.DataLoader(
    SVHNDataset(train_path, train_label,
                transforms.Compose([
                    transforms.Resize((64, 128)),
                    transforms.RandomCrop((60, 120)),
                    transforms.ColorJitter(0.3, 0.3, 0.2),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])),
    batch_size=40,
    shuffle=True,
    num_workers=10,
)

val_path = glob.glob('../../../dataset/tianchi_SVHN/val/*.png')
val_path.sort()
val_json = json.load(open('../../../dataset/tianchi_SVHN/val.json'))
val_label = [val_json[x]['label'] for x in val_json]
print(len(val_path), len(val_label))

val_loader = torch.utils.data.DataLoader(
    SVHNDataset(val_path, val_label,
                transforms.Compose([
                    transforms.Resize((60, 120)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])),
    batch_size=40,
    shuffle=False,
    num_workers=10,
)
```

```
30000 30000
10000 10000
```


## 定义分类模型

这里使用ResNet18模型进行特征提取

```python
class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()

        model_conv = models.resnet18(pretrained=True)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv

        self.fc1 = nn.Linear(512, 11)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 11)
        self.fc4 = nn.Linear(512, 11)
        self.fc5 = nn.Linear(512, 11)

    def forward(self, img):
        feat = self.cnn(img)
        # print(feat.shape)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        return c1, c2, c3, c4, c5
```

```python
def train(train_loader, model, criterion, optimizer):
    # 切换模型为训练模式
    model.train()
    train_loss = []

    for i, (input, target) in enumerate(train_loader):
        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        c0, c1, c2, c3, c4 = model(input)
        loss = criterion(c0, target[:, 0]) + \
                criterion(c1, target[:, 1]) + \
                criterion(c2, target[:, 2]) + \
                criterion(c3, target[:, 3]) + \
                criterion(c4, target[:, 4])

        # loss /= 6
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
    return np.mean(train_loss)

def validate(val_loader, model, criterion):
    # 切换模型为预测模型
    model.eval()
    val_loss = []

    # 不记录模型梯度信息
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if use_cuda:
                input = input.cuda()
                target = target.cuda()

            c0, c1, c2, c3, c4 = model(input)
            loss = criterion(c0, target[:, 0]) + \
                    criterion(c1, target[:, 1]) + \
                    criterion(c2, target[:, 2]) + \
                    criterion(c3, target[:, 3]) + \
                    criterion(c4, target[:, 4])
            # loss /= 6
            val_loss.append(loss.item())
    return np.mean(val_loss)

def predict(test_loader, model, tta=10):
    model.eval()
    test_pred_tta = None

    # TTA 次数
    for _ in range(tta):
        test_pred = []

        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                if use_cuda:
                    input = input.cuda()

                c0, c1, c2, c3, c4 = model(input)
                if use_cuda:
                    output = np.concatenate([
                        c0.data.cpu().numpy(),
                        c1.data.cpu().numpy(),
                        c2.data.cpu().numpy(),
                        c3.data.cpu().numpy(),
                        c4.data.cpu().numpy()], axis=1)
                else:
                    output = np.concatenate([
                        c0.data.numpy(),
                        c1.data.numpy(),
                        c2.data.numpy(),
                        c3.data.numpy(),
                        c4.data.numpy()], axis=1)

                test_pred.append(output)

        test_pred = np.vstack(test_pred)
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta
```


## 训练与验证

```python
model = SVHN_Model1()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.001)
best_loss = 1000.0

use_cuda = True
if use_cuda:
    model = model.cuda()

for epoch in range(10):
    train_loss = train(train_loader, model, criterion, optimizer)
    val_loss = validate(val_loader, model, criterion)
    
    val_label = [''.join(map(str, x)) for x in val_loader.dataset.img_label]
    val_predict_label = predict(val_loader, model, 1)
    val_predict_label = np.vstack([
        val_predict_label[:, :11].argmax(1),
        val_predict_label[:, 11:22].argmax(1),
        val_predict_label[:, 22:33].argmax(1),
        val_predict_label[:, 33:44].argmax(1),
        val_predict_label[:, 44:55].argmax(1),
    ]).T
    val_label_pred = []
    for x in val_predict_label:
        val_label_pred.append(''.join(map(str, x[x!=10])))
    
    val_char_acc = np.mean(np.array(val_label_pred) == np.array(val_label))
    
    print('Epoch: {0}, Train loss: {1} \t Val loss: {2}'.format(epoch, train_loss, val_loss))
    print('Val Acc', val_char_acc)
    # 记录下验证集精度
    if val_loss < best_loss:
        best_loss = val_loss
        # print('Find better model in Epoch {0}, saving model.'.format(epoch))
        torch.save(model.state_dict(), './model.pt')
```

```
Epoch: 0, Train loss: 3.5854218196868897 	Val loss: 3.6840851964950563   Val Acc 0.3205
Epoch: 1, Train loss: 2.26186142206192 	    Val loss: 3.3972847995758055   Val Acc 0.395
Epoch: 2, Train loss: 1.8909126870632171 	Val loss: 2.751760967731476    Val Acc 0.4849
Epoch: 3, Train loss: 1.686914145708084 	Val loss: 2.7759255743026734   Val Acc 0.4846
Epoch: 4, Train loss: 1.53107564496994 	    Val loss: 2.786783583164215    Val Acc 0.4913
Epoch: 5, Train loss: 1.418904512643814 	Val loss: 2.6081195278167724   Val Acc 0.5183
Epoch: 6, Train loss: 1.3347066225210826 	Val loss: 2.4557434034347536   Val Acc 0.5468
Epoch: 7, Train loss: 1.237933918039004 	Val loss: 2.576781086444855    Val Acc 0.5155
Epoch: 8, Train loss: 1.1775057731866836 	Val loss: 2.5613928298950195   Val Acc 0.5506
Epoch: 9, Train loss: 1.1077081708113352 	Val loss: 2.59310835814476     Val Acc 0.5388
```


## 预测并生成提交文件

```python
test_path = glob.glob('../../../dataset/tianchi_SVHN/test_a/*.png')
test_path.sort()
test_label = [[1]] * len(test_path)
print(len(test_path), len(test_label))

test_loader = torch.utils.data.DataLoader(
    SVHNDataset(test_path, test_label,
                transforms.Compose([
                    transforms.Resize((64, 128)),
                    transforms.RandomCrop((60, 120)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])),
    batch_size=40,
    shuffle=False,
    num_workers=10,
)
```

```
40000 40000
```

```python
# 加载保存的最优模型
model.load_state_dict(torch.load('model.pt'))

test_predict_label = predict(test_loader, model, 1)
print(test_predict_label.shape)

test_label = [''.join(map(str, x)) for x in test_loader.dataset.img_label]
test_predict_label = np.vstack([
    test_predict_label[:, :11].argmax(1),
    test_predict_label[:, 11:22].argmax(1),
    test_predict_label[:, 22:33].argmax(1),
    test_predict_label[:, 33:44].argmax(1),
    test_predict_label[:, 44:55].argmax(1),
]).T

test_label_pred = []
for x in test_predict_label:
    test_label_pred.append(''.join(map(str, x[x!=10])))

import pandas as pd
df_submit = pd.read_csv('../../../dataset/tianchi_SVHN/test_A_sample_submit.csv')
df_submit['file_code'] = test_label_pred
df_submit.to_csv('submit.csv', index=None)  
```

```
(40000, 55)
```


---
**Task01 预备知识 **

--- ***By: 安晟***


>可添加个人相关网址：博客、知乎、github等


**关于Datawhale**：

>Datawhale是一个专注于数据科学与AI领域的开源组织，汇集了众多领域院校和知名企业的优秀学习者，聚合>    了一群有开源精神和探索精神的团队成员。Datawhale以“for the learner，和学习者一起成长”为愿景，鼓励>    真实地展现自我、开放包容、互信互助、敢于试错和勇于担当。同时Datawhale 用开源的理念去探索开源内容>    、开源学习和开源方案，赋能人才培养，助力人才成长，建立起人与人，人与知识，人与企业和人与未来的联>    结。
