# 数据读取与数据扩增 

## 常见数据集简介   
    
### 1.ImageNet   
      
* #### 简介     
      
ImageNet项目是一个大型计算机视觉数据库，它按照WordNet层次结构（目前只有名词）组织图像数据，其中层次结构的每个节点都由成百上千个图像来描述，用于视觉目标识别软件研究。该项目已手动注释了1400多万张图像，以指出图片中的对象，并在至少100万张图像中提供了边框。ImageNet包含2万多个典型类别（synsets），例如大类别包括：amphibian、animal、appliance、bird、covering、device、fabric、fish等，每一类包含数百张图像。尽管实际图像不归ImageNet所有，但可以直接从ImageNet免费获得标注的第三方图像URL。2010年以来，ImageNet项目每年举办一次软件竞赛，即ImageNet大规模视觉识别挑战赛（ILSVRC）。

目前，ImageNet已广泛应用于图像分类(Classification)、目标定位(Object localization)、目标检测(Object detection)、视频目标检测(Object detection from video)、场景分类(Scene classification)、场景解析(Scene parsing)。    

<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.1_dataloader_and_augmentation/imageNet展示.png">

* #### 总览

  * Total number of non-empty synsets: 21841          
  * Total number of images: 14,197,122    
  * Number of images with bounding box annotations: 1,034,908 
  * Number of synsets with SIFT features: 1000    
  * Number of images with SIFT features: 1.2 million  

* #### 层次结构及下载方式  

 下图展示了ImageNet的层次结构：                               
                       
<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.1_dataloader_and_augmentation/imageNet层次结构.png">
               
 ImageNet有5种下载方式，如下图所示：                   
                       
<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.1_dataloader_and_augmentation/imageNet下载方式.png">

   *  所有原始图像可通过url下载：http://image-net.org/download-imageurls
   *  直接下载原始图像：需要自己申请注册一个账号，然后登录访问，普通邮箱（非组织和学校）无法获取权限。对于希望将图像用于非商业研究或教育目的的研究人员，可以在特定条件下通过ImageNet网站提供访问权限。
   *  下载图像sift features：http://image-net.org/download-features
   *  下载Object Bounding Boxes：http://image-net.org/download-bboxes
   *  下载Object Attributes： http://image-net.org/download-attributes       
                       
 官网：http://image-net.org/download-attributes
      
---
    
### 2.CIFAR-10

* #### 简介
    
CIFAR-10是一个小型图片分类数据集，该数据集共有60000张彩色图像，图像尺寸为32 * 32，共分为10个类，每类6000张图像。CIFAR-10数据集被分为5个训练的batch和一个测试的batch,每个batch中均包含10000张图像。测试批的数据里，取自10类中的每一类，每一类随机取1000张。抽剩下的就随机排列组成了训练批。注意一个训练批中的各类图像并不一定数量相同，总的来看训练批，每一类都有5000张图。

以下是数据集中的类，以及每个类中的10张随机图像：       

<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.1_dataloader_and_augmentation/CIFAR10展示.png">

值得说明的是这10类都是各自独立的，不会出现重叠，例如汽车并不包括卡车。

* #### 下载
官方给出了多个CIFAR-10数据集的版本：           

  Python版：[CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)       
  Matlab版：[CIFAR-10 Matlab version](https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz)      
  二进制版：[CIFAR-10 binary version (suitable for C programs)](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz)   
  
 * #### 数据集官网：https://www.cs.utoronto.ca/~kriz/cifar.html 
       
---           
          
### 3.MNIST
        
* #### 简介   

MNIST数据集(Mixed National Institute of Standards and Technology database)是美国国家标准与技术研究院收集整理的大型手写数字数据库。包含60,000个示例的训练集以及10,000个示例的测试集，其中训练集 (training set) 由来自 250 个不同人手写的数字构成, 其中 50% 是高中学生, 50% 来自人口普查局 (the Census Bureau) 的工作人员，测试集(test set) 也是同样比例的手写数字数据。可以说，完成MNIST手写数字分类和识别是计算机视觉领域的"Hello World"。        

<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.1_dataloader_and_augmentation/MNIST展示.png">

如下图所示，MNIST数据集的图像尺寸为28 * 28，且这些图像只包含灰度信息，灰度值在0~1之间。     

<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.1_dataloader_and_augmentation/MNIST展示2.png">

* #### 下载

  * [train-images-idx3-ubyte.gz:](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz)   training set images (9912422 bytes) 
  * [train-labels-idx1-ubyte.gz:](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz)  test set images (1648877 bytes) 
  * [t10k-images-idx3-ubyte.gz:](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz) test set images (1648877 bytes)
  * [t10k-labels-idx1-ubyte.gz:](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz)  test set labels (4542 bytes)

* #### 数据集官网：http://yann.lecun.com/exdb/mnist/
    
---    
     
### 4.PASCAL VOC
              
* #### 简介 
                  
PASCAL VOC为图像分类与物体检测提供了一整套标准的的数据集，并从2005年到2012年每年都举行一场图像检测竞赛。PASCAL全称为Pattern Analysis, Statical Modeling and Computational Learning，其中常用的数据集主要有VOC2007与VOC2012两个版本，VOC2007中包含了9963张标注过的图片以及24640个物体标签。在VOC2007之上，VOC2012进一步升级了数据集，一共11530张图片，包括人类；动物（鸟、猫、牛、狗、马、羊）；交通工具（飞机、自行车、船、公共汽车、小轿车、摩托车、火车）；室内（瓶子、椅子、餐桌、盆栽植物、沙发、电视）20个物体类别，图片尺寸为500x375。VOC整体图像质量较好，标注比较完整，非常适合模型的性能测试，比较适合做基线。     
      
<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.1_dataloader_and_augmentation/VOC展示.png">
         
 * ### 数据格式    
 ```
 .
└── VOCdevkit     #根目录
    └── VOC2012   #不同年份的数据集，这是2012的
        ├── Annotations        # 存放xml标签文件，与JPEGImages中的图片一一对应
        │     ├── 00001.xml 
        │     └── 00002.xml 
        ├── ImageSets          
        │   └── Main
        │     ├── train.txt    # txt文件中每一行包含一个图片的名称
        │     └── val.txt
        └── JPEGImages         # 存放源图片
              ├── 00001.jpg     
              └── 00002.jpg     
 ```
 
 * 官网： http://host.robots.ox.ac.uk/pascal/VOC/
      
---
      
## Pytorch数据读取方法简介  
   
在模型训练之前，我们需要先读取和加载数据，Pytorch的torchvision中已经包含了很多常用数据集，如Imagene，MNIST，CIFAR10、VOC等，利用torchvision可以很方便地读取;另外，在实际应用中，我们可能还需要从各种不同的数据集或自己构建的数据集中读取图像。所以，这一小节从常见数据集读取方法和自定义读取数据方法两个方面介绍Pytorch数据读取方法。      

本节以CIFAR10数据集为例进行介绍，默认将数据集下载在'Dive-into-CV-PyTorch/dataset/'目录下。    
      
### 1.常见数据集读取方法
    
对于常用的数据集，可以通过torchvision.datasets读取，所有datasets继承至torch.utils.data.Dataset，也就是说，它们实现了 __getitem__ 和 __len__ 方法。      
那么，pytorch支持哪些常用数据加载呢，可以参见：[torchvision.datasets](https://pytorch.org/docs/stable/torchvision/datasets.html)     
所有datasets读取方法的 __API__ 均十分相似，以CIFAR10为例：     
       
```python
torchvision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=False)
```      
参数： 
* root：存放数据集的路径。
* train（bool，可选）–如果为True，则从训练集创建数据集，否则从测试集创建。
* transform：数据预处理(数据增强)，如transforms.RandomRotation。
* target_transform：标注的预处理。    
* download：是否下载，若为True则从互联网下载，如果已经在root已经存在，就不会再次下载。    
      
#### 为了直观地体现数据读取方法，给出以下两个示例：      
      
* #### 读取示例1(从网上自动下载)        
          
```python
from PIL import Image
import torch
import torchvision 
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms         
      
train_data=torchvision.datasets.CIFAR10('../../../dataset', 
                                                      train=True, 
                                                      transform=None,  
                                                      target_transform=None, 
                                                      download=True)          
     
test_data=torchvision.datasets.CIFAR10('../../../dataset', 
                                                      train=False, 
                                                      transform=None, 
                                                      target_transform=None, 
                                                      download=True)      
```

* #### 读取示例2(数据增强)  
      
transform指定导入数据集时需要进行何种变换操作，transform中有很多方便的数据增强方法，我们将在下一小节介绍，在这里我们使用了尺寸归一化，随机颜色变换、随机旋转、图像像素归一化等组合变换。       
              
```python
from PIL import Image
import torch
import torchvision 
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms        
      
train_data=torchvision.datasets.CIFAR10('../../../dataset', train=True, 
                           transform=transforms.transforms.Compose([
                                           # 尺寸归一化
                                         transforms.Resize((64, 128)),
                                           # 随机颜色变换
                                         transforms.ColorJitter(0.2, 0.2, 0.2),
                                           # 加入随机旋转
                                         transforms.RandomRotation(5),
                                           # 对图像像素进行归一化
                                         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                      ]), 
                                      target_transform=None, download=False)          
     
```    
* #### 读取示例3(示例1+并行加载多个样本)         
            
数据下载完成后，我们还需要做数据装载操作，加快我们准备数据集的速度。datasets继承至torch.utils.data.Dataset，而torch.utils.data.DataLoader对Dataset进行了封装，所以我们可以利用DataLoader进行多线程批量读取。 
        
```python
from PIL import Image
import torch
import torchvision 
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms     
     
train_data=torchvision.datasets.CIFAR10('../../../dataset', train=True, 
                                                      transform=None,  
                                                      target_transform=None, 
                                                      download=True)          
       
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=2,
                                           shuffle=True,
                                           num_workers=4)       
                                                      
```        
       
batch_size设置了批量大小，shuffle设置为True在装载过程中为随机乱序，num_workers>=1表示多线程读取数据，在Win下num_workers最好设置为0。
        
### 2.自定义读取数据方法 

Pytorch自定义读取数据的方式 主要涉及到两个类：
         
* torch.utils.data.Dataset
* torch.utils.data.DataLoader    
    
你可能会问有了Dataset和DataLoder究竟有何区别？其实这两个是两个不同的概念，是为了实现不同的功能。

* Dataset：对数据集的封装，提供索引方式的对数据样本进行读取
* DataLoder：对Dataset进行封装，提供批量读取的迭代读取      
      
 
想要读取我们自己数据集中的数据，就需要写一个Dataset的子类，并对 __getitem__ 和  __len__  方法进行实现。下面我们看一下构建Dataset类的基本结构：    
    
 ```python
 
from torch.utils.data.dataset import Dataset
class MyDataset(Dataset):#继承Dataset
    def __init__(self):
        #  初始化文件路径或文件名列表。
        pass
    def __getitem__(self, index):

         ＃1。从文件中读取一个数据（例如，使用numpy.fromfile，PIL.Image.open，cv2.imread）。
         ＃2。预处理数据（例如torchvision.Transform）。
         ＃3。返回数据对（例如图像和标签）。
        pass
    def __len__(self):
        return count     
 ```          
    
 *  __init__() : 初始化模块，初始化该类的一些基本参数。
 * __getitem__() : 接收一个index，这个index通常指的是一个list的index，这个list的每个元素就包含了图片数据的路径和标签信息,返回数据对（图像和标签）。
 * __len__() : 返回所有数据的数量。     
       
当我们根据数据集的模式构建好MyDataset后，同样也就可以利用DataLoader进行多线程批量读取啦。    
       
这里以[SVHN](http://ufldl.stanford.edu/housenumbers/)数据集为例构建了一个SVHNDataset完成对SVHN数据集的读取，这里仅是截取了天池CV入门赛baseline代码中的一段作为示例，详细内容可参考：[datawhale_team_learning](https://github.com/datawhalechina/team-learning/blob/master/03%20%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5%EF%BC%88%E8%A1%97%E6%99%AF%E5%AD%97%E7%AC%A6%E7%BC%96%E7%A0%81%E8%AF%86%E5%88%AB%EF%BC%89/Datawhale%20%E9%9B%B6%E5%9F%BA%E7%A1%80%E5%85%A5%E9%97%A8CV%20-%20Task%2002%20%E6%95%B0%E6%8D%AE%E8%AF%BB%E5%8F%96%E4%B8%8E%E6%95%B0%E6%8D%AE%E6%89%A9%E5%A2%9E.md)。

```python
import os, sys, glob, shutil, json
import cv2

from PIL import Image
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

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
        
        # 原始SVHN中类别10为数字0
        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl)  + (5 - len(lbl)) * [10]
        
        return img, torch.from_numpy(np.array(lbl[:5]))

    def __len__(self):
        return len(self.img_path)

train_path = glob.glob('../dataset/*.png')
train_path.sort()
train_json = json.load(open('../input/train.json'))
train_label = [train_json[x]['label'] for x in train_json]

data = SVHNDataset(train_path, train_label,
          transforms.Compose([
              # 缩放到固定尺寸
              transforms.Resize((64, 128)),
              # 将图片转换为pytorch 的tesntor
              transforms.ToTensor(),
              # 对图像像素进行归一化
              transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ]))
```
     
---   
     
## 图像分类常见数据扩增方法       
       
在常见的数据扩增方法中，一般会从图像颜色、尺寸、形态、亮度/对比度、噪声和像素等角度进行变换。当然不同的数据扩增方法可以自由进行组合，得到更加丰富的数据扩增方法。    
    
以torchvision为例，常用的数据增强的函数主要集成在了transforms中,这里列出19种图像扩增强方法：    
      
### 1.裁剪      
 * transforms.CenterCrop   ——— 对图片中心进行裁剪        
 * transforms.RandomCrop  ——— 随机区域裁剪
 * transforms.RandomResizeCrop ——— 随机长宽比裁剪
 * transforms.FiveCrop ——— 对图像四个角和中心进行裁剪得到五分图像
 * transforms.TenCrop ——— 上下左右中心裁剪后翻转
  
### 2.翻转和旋转    
 * transforms.RandomHorizontalFlip ——— 依概率随机水平翻转
 * transforms.RandomVerticalFlip ——— 依概率随机垂直翻转
 * transforms.RandomRotation ——— 随机旋转  
      
### 3. 图像变换
 * transforms.Pad ——— 使用固定值进行像素填充
 * transforms.ColorJitter ——— 对图像颜色的对比度、饱和度和亮度进行变换    
 * transforms.Grayscale ——— 对图像进行灰度变换
 * transforms.RandomGrayscale ——— 依概率灰度化
 * transforms.RandomAffine ——— 随机仿射变换
 * transforms.LinearTransformation ——— 线性变换
 * transforms.RandomErasing ——— 随机选择图像中的矩形区域并擦除其像素
 * transforms.Lambda   ——— 用户自定义变换
 * tuxiang transforms.Resize  ——— 尺度缩放
 * transforms.Totensor  ——— 将 PIL Image 或者 numpy.ndarray 格式的数据转换成 tensor
 * transforms.Normalize ——— 图像标准化
     
### 部分上述图像变换的代码示例和效果     
    
#### 导入包和读入图像
     
```python
import os, sys, glob, shutil, json
import numpy as np
import cv2

from PIL import Image
import torch
import torchvision 
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms  

outfile='../../../dataset'      
im = Image.open('../../../dataset/*.png')         
```
      
<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.1_dataloader_and_augmentation/cat.png">
             
#### 一.裁剪    
      
#### 1.中心裁剪：transforms.CenterCrop
```python
new_im = transforms.CenterCrop([200,200])(im)  
```
      
<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.1_dataloader_and_augmentation/CenterCrop.png">
      
#### 2.随机裁剪：：transforms.RandomCrop
```python
new_im =transforms.RandomCrop([200,200])(im) 
```     
     
<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.1_dataloader_and_augmentation/RandomCrop.png">
        
#### 3.随机长宽比裁剪 transforms.RandomResizedCrop
```python 
new_im =transforms.RandomResizedCrop(200, 
                             scale=(0.08, 1.0), 
                             ratio=(0.75, 1.55), 
                             interpolation=2)(im) 
```    
      
<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.1_dataloader_and_augmentation/RandomResizedCrop.png">
           
#### 二.翻转和旋转 
      
#### 4.依概率p水平翻转：transforms.RandomHorizontalFlip     
```python
new_im =transforms.RandomHorizontalFlip(0.7)(im) 
```    
        
<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.1_dataloader_and_augmentation/RandomHorizontalFlip.png">
        
#### 5.依概率p垂直翻转：transforms.RandomVerticalFlip 
```python
new_im=transforms.RandomVerticalFlip(0.8)(im)
```   
     
<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.1_dataloader_and_augmentation/RandomVerticalFlip.png">
      
#### 6.随机旋转：transforms.RandomRotation    
```python
new_im=transforms.RandomRotation(30)(im)    
```     
      
<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.1_dataloader_and_augmentation/RandomRotation.png">
     
#### 三.图像变换   
    
#### 7.填充：transforms.Pad      
```python     
 new_im =transforms.Pad(10, fill=0, padding_mode='constant')(im)   
```     
       
<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.1_dataloader_and_augmentation/Pad.png">

#### 8.调整亮度、对比度和饱和度：transforms.ColorJitter 
```python    
new_im=transforms.ColorJitter(brightness=1,       
                              contrast=0.5,       
                              saturation=0.5,    
                              hue=0.4)(im)        
```
      
<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.1_dataloader_and_augmentation/ColorJitter.png">
     
#### 9.转灰度图：transforms.Grayscale
```python
new_im=transforms.Grayscale(1)(im)      
```   
      
<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.1_dataloader_and_augmentation/Grayscale.png">
       
#### 10. 仿射变换：transforms.RandomAffine
```python 
new_im =transforms.RandomAffine(45,(0.5,0.7),(0.8,0.5),3)(im) 
```    
     
<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.1_dataloader_and_augmentation/RandomAffine.png">
      
#### 11.尺寸缩放:transforms.Resize   
```python 
new_im=transforms.Resize([100,200])(im)      
```   
      
<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.1_dataloader_and_augmentation/Resize.png">
       
#### 12.转Tensor、标准化和转换为PILImage    
```python     
mean = [0.45, 0.5, 0.5]   
std = [0.3, 0.6, 0.5]    
transform = transforms.Compose([
    transforms.ToTensor(), #转Tensor
    transforms.Normalize(mean, std), 
    transforms.ToPILImage() # 这里是为了可视化，故将其再转为 PIL
])     
new_img = transform(im)   
```
        
<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.1_dataloader_and_augmentation/Normalize.png">
              
#### 图像显示和保存
```python
new_im.show()         
new_im.save(os.path.join(outfile, '*.png'))
```
      
---              
      
## 读取数据并进行数据扩增示例      
     
前文对数据读取和数据扩增方法进行了介绍，那么这一部分就结合数据读取和数据扩增给出一个数据加载的示例。  
    
 
#### 示例1   
     
以CIFAR10数据集为例，通过常见数据集读取的方法加载数据。       
    
```python      
import os, sys, glob, shutil, json
import numpy as np
import cv2

from PIL import Image
import torch
import torchvision 
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms  


transform = transforms.Compose([
                       transforms.Resize((100, 100)),
                       transforms.ColorJitter(0.3, 0.3, 0.2),
                       transforms.RandomRotation(10),
                       transforms.RandomAffine(10,(0.5,0.7),(0.8,0.5),0.2),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data=torchvision.datasets.CIFAR10('../../../dataset', train=True, 
                                        transform= transform, 
                                        target_transform=None, 
                                        download=False)

test_data=torchvision.datasets.CIFAR10('../../../dataset', train=False, 
                                        transform= transform, 
                                        target_transform=None, 
                                        download=False)

train_loader = torch.utils.data.DataLoader(train_data, 
                                            batch_size=64,
                                            shuffle=True,
                                            num_workers=4)

test_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=64,
                                          shuffle=True,
                                          num_workers=4)
```
         
#### 示例2 
     
以SVHN数据集数据集为例，通过重写Dataset类的方法加载数据，详细内容可参考：[datawhale_team_learning](https://github.com/datawhalechina/team-learning/blob/master/03%20%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E5%AE%9E%E8%B7%B5%EF%BC%88%E8%A1%97%E6%99%AF%E5%AD%97%E7%AC%A6%E7%BC%96%E7%A0%81%E8%AF%86%E5%88%AB%EF%BC%89/Datawhale%20%E9%9B%B6%E5%9F%BA%E7%A1%80%E5%85%A5%E9%97%A8CV%20-%20Task%2002%20%E6%95%B0%E6%8D%AE%E8%AF%BB%E5%8F%96%E4%B8%8E%E6%95%B0%E6%8D%AE%E6%89%A9%E5%A2%9E.md)。。   
     
```python
import os, sys, glob, shutil, json
import cv2

from PIL import Image
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

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
        
        # 原始SVHN中类别10为数字0
        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl)  + (5 - len(lbl)) * [10]
        
        return img, torch.from_numpy(np.array(lbl[:5]))

    def __len__(self):
        return len(self.img_path)

# 假设你的数据集根目录放在了dataset下
train_path = glob.glob('../../../dataset/SVHN/train/*.png')
train_path.sort()
train_json = json.load(open('../../../dataset/SVHN/train.json'))
train_label = [train_json[x]['label'] for x in train_json]

train_loader = torch.utils.data.DataLoader(
        SVHNDataset(train_path, train_label,
                   transforms.Compose([
                       transforms.Resize((64, 128)),
                       transforms.ColorJitter(0.3, 0.3, 0.2),
                       transforms.RandomRotation(5),
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])), 
    batch_size=10, # 每批样本个数
    shuffle=False, # 是否打乱顺序
    num_workers=10, # 读取的线程个数
)

for data in train_loader:
    break
```
        
## 总结     

本节对4个常用数据集进行了简单介绍，并讲解了利用torchvision的数据集读取方法，然后介绍了常见的数据增强方法且展示了实现代码和效果，最后结合数据集读取和数据扩增给出了两种数据加载示例。  

---    
     
贡献者：

--- ***By: 小武***

>https://blog.csdn.net/weixin_40647819

--- ***By: 阿水***

>微信公众号：Coggle数据科学

      
**关于Datawhale**：     
      
>Datawhale是一个专注于数据科学与AI领域的开源组织，汇集了众多领域院校和知名企业的优秀学习者，聚合了一群有开源精神和探索精神的团队成员。Datawhale以“for the learner，和学习者一起成长”为愿景，鼓励真实地展现自我、开放包容、互信互助、敢于试错和勇于担当。同时Datawhale 用开源的理念去探索开源内容、开源学习和开源方案，赋能人才培养，助力人才成长，建立起人与人，人与知识，人与企业和人与未来的联结。
