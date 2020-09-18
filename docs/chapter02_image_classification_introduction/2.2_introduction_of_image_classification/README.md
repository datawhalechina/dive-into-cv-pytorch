# 经典图像分类模型介绍


## 介绍

​		本文我们来回顾经典的卷积神经网络（Convolution Nerual Network，简称CNN ）。CNN是一类特殊的人工神经网络，是深度学习中重要的一个分支。CNN在很多领域都表现优异，精度和速度比传统计算学习算法高很多。特别是在计算机视觉领域，CNN是解决图像分类、图像检索、物体检测和语义分割的主流模型。

​		学习本章节内容前，希望读者已经了解多层感知机，以及反向传播算法等原理。这里我们希望进行更多直觉上与工程上的讲解，因此不会涉及太多理论公式。首先回顾多层感知机（MLP），如下左图[^1]的例子，这个网络可以完成简单的分类功能。怎么实现呢？每个实例从输入层（input layer）输入，因为输入维度为3，所以要求输入实例有三个维度。接下来，通过隐藏层（hidden layer）进行**升维**，网络层与层之间采用全连接（fully connected）的方式，每一层输出都要通过**激活函数**进行非线性变换，前向计算得到输出结果（output layer）。训练采用有监督的学习方式进行梯度反向传播（BP）。

<center>
  <div>
  <img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.2_introduction_of_image_classification/cnn.jpeg" width="40%" />
  <img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.2_introduction_of_image_classification/cnn1.jpeg" style="border-left: 1px solid black;" width="48%" />
  </div>
  <div>
    左：具有4层的感知机，右：卷积神经网络
  </div>
</center>

​		MLP能够对简单的，维度较低的数据进行分类。而对于维度较高的图片，便凸显问题。例如，cifar10数据集每张图都是$32 \times 32$的图片，如果我们用一个MLP网络进行图像分类，其输入是$32 \times 32 \times 3 = 3072$维，假设这是一个十分类的MLP网络，其架构是`3072 --> 4096 --> 4096--> 1`0 ,网络的参数为
$$
3072 \times 4096 + 4096 \times 4096 + 4096 \times 10 = 29401088 \approx 三千万
$$
​		小小一张图片需要耗费巨大参数，如果将图片换成现在人们常用的图片，参数量更是惊人的！于是，CNN很好地解决了这个问题，网络的每层都只有三个维度：宽，高，深度。这里的深度指图像通道数，每个通道都是图片，代表我们要分析的一个属性。比如，灰度图通道数是1，RGB图像通道数是3，CMYK[^2]图像通道数是4，而卷积网络层的通道数会更高。


## 卷积神经网络基础

​		下面我们对CNN中的关键知识做介绍。在学习卷积神经网络时，笔者建议先直观理解思想，再研究原理。现在有许多CNN计算可视化工具，我们可以借助这些工具来学习CNN。这里我们使用最近新出的[CNN explainer](https://poloclub.github.io/cnn-explainer/)[^2] 来学习这些思想，CNN explainer使用了一个[tiny-VGG](https://github.com/poloclub/cnn-explainer/tree/master/tiny-vgg)，在浏览器里，我们可以看到CNN计算的细节。


<center>
  <div>
    <img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.2_introduction_of_image_classification/overview.png" style="zoom:30%;" />
  </div>
  <div>
  	CNN explainer
  </div>
</center>

### 输入层
​		输入层（最左边的层）代表输入到CNN中的图像。 因为RGB图像作为输入，所以输入层具有三个通道，分别对应于该层中显示的红色，绿色和蓝色通道。

### 二维卷积层

#### 卷积（convolution）

​		二维卷积层的参数由一组可学习的**卷积核**（filter）组成。 每个卷积核尺度都很小（沿宽度和高度方向），但是深度会延伸到输入**感受野**（receptive field 又译接收域）的所有通道，也就是说，卷积核和感受野的尺寸大小是一样的，尺寸对应的。

​		卷积核和感受野之间的卷积操作如下图。感受野与卷积核大小一致，对应位置相乘再相加，即可得到结果。可见：单个卷积核与图片的一个感受野进行卷积，结果是下一层图片对应位置的一个像素点，一张输入图片与一个卷积核的输出结果是一张图片。

<div align="center">
	<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.2_introduction_of_image_classification/conv.png" style="zoom:50%;" />
	<div> 卷积操作</div>
</div>
​		**多输入通道** 卷积核窗口形状为$k_h\times k_w$。当$c_i=1$时，我们知道卷积核只包含一个形状为$k_h\times k_w$的二维数组。当$c_i > 1$时，我们将会为每个输入通道各分配一个形状为$k_h\times k_w$的核数组。把这$c_i$个数组在输入通道维上连结，即得到一个形状为$c_i\times k_h\times k_w$的卷积核。如下图，点开第一个卷积层，可以看到一组通道数为3的卷积核，与输入的通道为3的图像进行卷积，得到3个中间结果（表示在intermidiate层），再将三个中间结果对象像素位置相加，加上可学习的bias，得到一个通道的卷积结果。

​		**多输出通道** 当输入通道有多个时，因为我们对各个通道的结果做了累加，所以不论输入通道数是多少，输出通道数总是为1。设卷积核输入通道数和输出通道数分别为$c_i$和$c_o$，高和宽分别为$k_h$和$k_w$。如果希望得到含多个通道的输出，我们可以为每个输出通道分别创建形状为$c_i\times k_h\times k_w$的核数组。将它们在输出通道维上连结，卷积核的形状即$c_o\times c_i\times k_h\times k_w$。如果我们希望得到多个通道的输出，输入层有3个神经元，而后继的conv_1层有10个神经元，因此，我们需要10组卷积核，一共是$10 \times 3 = 30$个卷积核。

**总结**

- 多输入通道需要一**组**卷积核进行卷积操作，得到一个通道输出
- 多输出通道需要多**组**卷积核进行卷积操作，得到多个通道输出



<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.2_introduction_of_image_classification/convlayer_overview_demo.gif" width="60%" height="60%" />
  <div>
    卷积层
  </div>
</div>
​		每个卷积核有9个参数，计算一个卷积层参数大小则为：
$$
(3 \times 3 )\times 30 + 10 = 280
$$
​		相比于全连接，参数数量大大减小。



#### 填充（padding）

​		填充可以在激活图的边界处保存数据，从而提高性能，并且可以帮助保留输入的空间大小，从而使体系结构设计人员可以构建性能更高，更流畅的网络。 存在许多填充技术，但是最常用的方法是零填充，因为它的性能，简单性和计算效率高。 该技术涉及在输入的边缘周围对称地添加零。 许多高性能的CNN（例如AlexNet）都采用了这种方法。

#### 步长（itride）

​		步幅表示卷积核一次应移动多少像素。如上面的卷积层例子，Tiny VGG的卷积层使用步幅为1，这意味着在输入的3x3窗口上执行点积以产生输出值，然后将其移至 每进行一次后续操作，就增加一个像素。 跨度对CNN的影响类似于内核大小。 随着步幅的减小，由于提取了更多的数据，因此可以了解更多的功能，但输出层也更大。 相反，随着步幅的增加，这将导致特征提取更加受限，输出层尺寸更小。 网络设计人员的职责之一是在实现CNN时确保内核对称地跨输入滑动。 

### 激活函数

#### Relu

​		CNN包含大量的图层，这些图层能够学习到越来越多的功能。为什么CNN能取得如此大的准确性，其原因在于它们的非线性。非线性是产生非线性决策边界所必需的，因此输出不能写为输入的线性组合。如果没有非线性激活函数，那么CNN架构将演变为一个等效的卷积层，其性能就不会变得那么好。经验上表明，ReLU相比其他激活函数，比如Sigmoid函数，前者性能会更好。Relu计算方法很简单：
$$
\text{ReLU}(x) = \max(0,x)
$$

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.2_introduction_of_image_classification/relu_graph.svg" width="30%" height="30%" />
  <div>
    卷积层
  </div>
</div>

#### Softmax

softmax操作的主要目的是：确保CNN输出的总和为1。因此，softmax操作可用于将模型输出缩放为概率。
$$
\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
$$


<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.2_introduction_of_image_classification/softmax_animation.gif"  />
  <div>
    Softmax
  </div>
</div>

### 池化层

​		在不同的CNN架构中，池化层的类型很多，但是它们的目的都是要逐渐**减小网络的空间范围**，从而减少网络的参数和总体计算。 上面的Tiny VGG架构中使用的池类型为最大池化(Max-Pooling)。

​		最大池化操作需要在设计网络过程中选择过滤核的大小和步长。 一旦选定，该操作将以指定的步长在输入上滑动过滤核，同时仅从输入中选择每个内核切片上的最大值以产生输出值。 在上面的Tiny VGG体系结构中，池化层使用2x2过滤核，步长为2。使用这些规范进行此操作，将导致75％的激活被丢弃。 通过丢弃如此多的值，Tiny VGG的计算效率更高，并且避免了**过拟合**。

## 经典图像分类模型介绍

#### LeNet

 <center><img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.2_introduction_of_image_classification/lenet.png" alt="IMG" style="zoom:100%;" /></center>   

##### 网络架构

LeNet分为卷积层块和全连接层块两个部分。下面我们分别介绍这两个模块。

卷积层块里的基本单位是卷积层后接最大池化层：卷积层用来识别图像里的空间模式，如**线条**和**物体局部**，之后的最大池化层则用来降低卷积层对位置的敏感性。卷积层块由两个这样的基本单位重复堆叠构成。在卷积层块中，每个卷积层都使用5×5的窗口，并在输出上使用sigmoid激活函数。第一个卷积层输出通道数为6，第二个卷积层输出通道数则增加到16。这是因为第二个卷积层比第一个卷积层的输入的高和宽要小，所以增加输出通道使两个卷积层的参数尺寸类似。卷积层块的两个最大池化层的窗口形状均为2×2，且步幅为2。由于池化窗口与步幅形状相同，池化窗口在输入上每次滑动所覆盖的区域互不重叠。

卷积层块的输出形状为(批量大小, 通道, 高, 宽)。当卷积层块的输出传入全连接层块时，全连接层块会将小批量中每个样本变平（flatten）。也就是说，全连接层的输入形状将变成二维，其中第一维是小批量中的样本，第二维是每个样本变平后的向量表示，且向量长度为通道、高和宽的乘积。全连接层块含3个全连接层。它们的输出个数分别是120、84和10，其中10为输出的类别个数。  

##### 代码实战

[完整代码](classical_cnn_models/lenet/LeNet.py)

网络定义:

```python
#Lenet network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) #in_channels, out_channels, kernel_size, stride=1 ...
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))

        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```



​    

##### 总结

通过多次卷积和池化，CNN的最后一层**将输入的图像像素映射为具体的输出**。如在分类任务中会转换为不同类别的概率输出，然后计算真实标签与CNN模型的预测结果的差异，并通过反向传播更新每层的参数，并在更新完成后再次前向传播，如此反复直到训练完成 。与传统机器学习模型相比，CNN具有一种端到端（End to End）的思路。在CNN训练的过程中是直接从图像像素到最终的输出，并不涉及到具体的特征提取和构建模型的过程，也不需要人工的参与。

#### AlexNet

在**AlexNet**之前，深度学习已经在语音识别和其它几个领域获得了一些关注，但正是通过这篇论文，计算机视觉群体开始重视深度学习，并确信深度学习可以应用于计算机视觉领域。此后，深度学习在计算机视觉及其它领域的影响力与日俱增。



##### 网络架构

 <center><img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.2_introduction_of_image_classification/alexnet.png" alt="IMG" style="zoom:100%;" /></center>  

​		在写这篇论文的时候，**GPU**的处理速度还比较慢，所以**AlexNet**采用了非常复杂的方法在两个**GPU**上进行训练。大致原理是，这些层分别拆分到两个不同的**GPU**上，同时还专门有一个方法用于两个**GPU**进行交流。

- 上下两个部分结构一样，为了方便在两块GPU上进行训练
- 每个部分有五个卷积层，三个全连接层

【注】由于上下两部分完全一致，分析时一般取一部分即可。



##### 代码实战

[完整代码](classical_cnn_models/AlexNet/AlexNet.py)

模型定义

```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x
```



##### 实验结果

​		在2010年的 ImageNet LSVRC-2010上，AlexNet 在给包含有1000种类别的共120万张高分辨率图片的分类任务中，在测试集上的top-1和top-5错误率为37.5%和17.0%（**top-5 错误率：即对一张图像预测5个类别，只要有一个和人工标注类别相同就算对，否则算错。同理top-1对一张图像只预测1个类别**），在 ImageNet LSVRC-2012 的比赛中，取得了top-5错误率为15.3%的成绩。AlexNet 有6亿个参数和650,000个神经元，包含5个卷积层，有些层后面跟了max-pooling层，3个全连接层，为了减少过拟合，在全连接层使用了dropout，下面进行更加详细的介绍。

​		数据来源于**[ImageNet](http://www.image-net.org/)**，训练集包含120万张图片，验证集包含5万张图片，测试集包含15万张图片，这些图片分为了1000个类别，并且有多种不同的分辨率，但是AlexNet的输入要求是固定的分辨率，为了解决这个问题，Alex的团队采用低采样率把每张图片的分辨率降为256×256，具体方法就是给定一张矩形图像，首先重新缩放图像，使得较短边的长度为256，然后从结果图像的中心裁剪出256×256大小的图片。

​		

##### 总结

- 使用relu。在此之前都用饱和的非线性激活函数$tanh(x)=(1+e^{-x})^{-1}$，但其比非饱和非线性函数$relu(x)=max(0,x)$函数，梯度下降慢，因此用了`relu`函数，结果如下图

 <center><div>
   <img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.2_introduction_of_image_classification/alex_test1.png" alt="IMG" style="zoom:40%;" />
   </div>
	<div>
    实线使用了relu，虚线使用了tanh
   </div>
</center>  



- 多GPU训练。
- 用Dropout来控制全连接层的模型复杂度。
- 引入数据增强，如翻转、裁剪和颜色变化，从而进一步扩大数据集来缓解过拟合。
- 相对复杂，包含大量超参数



#### VGG

[VGG](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1409.1556)是Oxford的**V**isual **G**eometry **G**roup的组提出的。该网络是在ILSVRC 2014上的相关工作，主要工作是证明了增加网络的深度能够在一定程度上影响网络最终的性能。VGG有两种结构，分别是VGG16和VGG19，两者并没有本质上的区别，只是网络深度不一样。[^3]

##### 网络架构

<center>
   <div>
     <img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.2_introduction_of_image_classification/VGG.png" alt="IMG" style="zoom:100%;" />
   </div>
   <div>
     VGG16
   </div>
</center>  

##### 代码实战

[简单vgg案例](./classical_cnn_models/VGG/main.py)

[VGG各模型综合比较](./classical_cnn_models/pytorch-vgg-cifar10/)

##### 总结

- 使用了3个3x3卷积核来代替7x7卷积核，使用了2个3x3卷积核来代替5*5卷积核。这样做的主要目的是在保证具有相同感知野的条件下，提升了网络的深度（因为多层非线性层可以增加网络深度来保证学习更复杂的模式），在一定程度上提升了神经网络的效果。

> 设输入通道数和输出通道数都为C， 3个步长为1的3x3卷积核的一层层叠加作用可看成一个大小为7的感受野（其实就表示3个3x3连续卷积相当于一个7x7卷积），其参数总量为$ 3\times (9\times C^2)$ ，如果直接使用7x7卷积核，其参数总量为 $49\times C^2$ 。很明显，$27\times C^2 $ 小于$49\times C^2$，即减少了参数；而且3x3卷积核有利于更好地保持图像性质。

- VGGNet的结构非常简洁，整个网络都使用了同样大小的卷积核尺寸（3x3）和最大池化尺寸（2x2）。
- 几个小滤波器（3x3）卷积层的组合比一个大滤波器（5x5或7x7）卷积层好：
- 验证了通过不断加深网络结构可以提升性能。

- 缺点是VGG耗费更多计算资源，并且使用了更多的参数，这里不是3x3卷积的原因，其中绝大多数的参数都是来自于第一个全连接层。

#### 网络中的网络（NiN）

​		前⼏节介绍的LeNet、AlexNet和VGG在设计上的共同之处是:先以由卷积层构成的模块充分抽取空间特征，再以由全连接层构成的模块来输出分类结果（下左图）。其中，AlexNet和VGG对LeNet的改进主要在于如何对这两个模块加宽(增加通道数)和加深。

​		本节我们介绍网络中的⽹络(NiN)。它提出了另外⼀个思路，即串联多个由卷积层和“**全连接**”层构成的⼩网络，又称MLP卷积，来构建⼀个深层网络（下右图）。先进行一次普通的卷积（比如3x3），紧跟再进行一次**1x1的卷积**，对于某个像素点来说1x1卷积等效于该像素点在所有特征上进行一次全连接的计算，所以右侧图的1x1卷积画成了全连接层的形式，需要注意的是NIN结构中无论是第一个3x3卷积还是新增的1x1卷积，后面都紧跟着激活函数（比如relu）。使用这种结构的原因有两个，一， MLP与CNN更兼容，并使用反向传播进行训练；二，MLP本身可以是深度模型，这与特性重用的精神是一致的。注意到这种1x1卷积方式是非常有效的，对后来的网络设计有非常大的启发，虽然NiN在后来应用不多，但1x1卷积的思想得到了广泛使用。

​		

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.2_introduction_of_image_classification/linear_conv.png" width="40%">
  <img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.2_introduction_of_image_classification/mlp_conv.png" style="border-left: 1px solid black;" width="40%">
</div>

​		

##### 网络架构

​		前3层是MLP卷积层，最后一层是全局平均池化。

<center>
   <div>
     <img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.2_introduction_of_image_classification/NiN.png" alt="IMG" style="zoom:100%;" />
   </div>
   <div>
     NiN网络架构
   </div>
</center>  
##### 代码实战

- 完整运行版[->](classical_cnn_models/NiN/NiN.py)
- 网络代码

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classifier = nn.Sequential(
          	#MLP卷积层1
            nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 96, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5),

          	#MLP卷积层2
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5),
						
          	#MLP卷积层3
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0),

        )

    def forward(self, x):
        x = self.classifier(x)
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        return x
```



##### 总结

- NiN重复使用由卷积层和代替全连接层的1×1卷积层构成的NiN块来构建深层网络
- 1x1卷积等效于该像素点在所有特征上进行一次全连接的计算，起到了**压缩通道**，即**降维**的作用，减少了通道的数量。
- NiN去除了容易造成过拟合的全连接输出层，而是将其替换成输出通道数等于标签类别数 的NiN块和全局平均池化层。
- NiN的以上设计思想影响了后面一系列卷积神经网络的设计。

#### 含并行连结的网络（GoogLeNet）

​		在2014年的ImageNet图像识别挑战赛中，一个名叫GoogLeNet的网络结构大放异彩 。它虽然在名字上向LeNet致敬，但在网络结构上已经很难看到LeNet的影子。GoogLeNet吸收了NiN中网络串联网络的思想，并在此基础上做了很大改进。在随后的几年里，研究人员对GoogLeNet进行了数次改进，本节将介绍这个模型系列的第一个版本。

​		

##### 网络架构



<div align=center>
<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.2_introduction_of_image_classification/googlenet.png"/>
</div>
<div align=center>Inception块的结构</div>

​		GoogLeNet中的基础卷积块叫作Inception块，得名于同名电影《盗梦空间》（Inception）。与上一节介绍的NiN块相比，这个基础块在结构上更加复杂。基本思想是**Inception**网络不需要人为决定使用哪个过滤器或者是否需要池化，而是由网络自行确定这些参数，你可以给网络添加这些参数的所有可能值，然后把这些输出连接起来，让网络自己学习它需要什么样的参数，采用哪些过滤器组合。

​		上图显示了Inception块的两个版本，图（a）是Inception块的基础版本，有4条并行的线路。前3条线路使用窗口大小分别是$1\times 1$、$3\times 3$和$5\times 5$的卷积层，第四条使用$3\times 3$最大池化层来抽取不同空间尺寸下的信息，再用1x1卷积改变通道数。

​		图（b）在中间2个线路会对输入先做$1\times 1$卷积来**减少输入通道数，以降低模型复杂度**。

​		4条线路**都使用了合适的填充来使输入与输出的高和宽一致**。最后我们将每条线路的输出在通道维上连结，并输入到接下来的层中去。

##### 代码实战

- 完整运行版[->](classical_cnn_models/GoogLeNet/main.py)

- 网络架构

```python
# https://github.com/facebookresearch/mixup-cifar10/blob/master/models/googlenet.py
'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class Inception(nn.Module):
    """
    1、输入通过Inception模块的4个分支分别计算，得到的输出宽和高相同(因为使用了padding)，而通道不同。
    2、将4个分支的通道进行简单的合并，即得到Inception模块的输出。
    3、每次卷积之后都使用批正则化`BatchNorm2d`，并使用relu函数进行激活。
    """

    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            # 2个3x3卷积代替1个5x5卷积
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
```

##### 总结

- 使用1x1的卷积，原因一是为了减少通道数，降低模型复杂度，二是为了提取更丰富的特征
- 在构建神经网络层的时候，不想决定池化层是使用1×1，3×3还是5×5的过滤器，那么**Inception**模块就是最好的选择。我们可以应用各种类型的过滤器，只需要把输出连接起来。

#### 批量归一化（Batch Normalization）

##### 为什么要进行批量归一化

​	   本节我们介绍批量归一化（batch normalization）层，它能让较深的神经网络的训练变得更加容易。通常来说，数据标准化预处理对于浅层模型就足够有效了。随着模型训练的进行，当每层中参数更新时，靠近输出层的输出较难出现剧烈变化。但对深层神经网络来说，即使输入数据已做标准化，训练中模型参数的更新依然很容易造成靠近输出层输出的剧烈变化。这种计算数值的不稳定性通常令我们难以训练出有效的深度模型。

​		批量归一化的提出正是为了应对深度模型训练的挑战。在模型训练时，批量归一化利用小批量上的均值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。**批量归一化和下一节将要介绍的残差网络为训练和设计深度模型提供了两类重要思路。**

##### 怎样进行批量归一化

- 对全连接层做批量归一化

我们先考虑如何对全连接层做批量归一化。通常，我们将批量归一化层置于全连接层中的仿射变换和激活函数之间。设全连接层的输入为$\boldsymbol{u}$，权重参数和偏差参数分别为$\boldsymbol{W}$和$\boldsymbol{b}$，激活函数为$\phi$。设批量归一化的运算符为$\text{BN}$。那么，使用批量归一化的全连接层的输出为
$$
\phi(\text{BN}(\boldsymbol{x})),
$$


其中批量归一化输入$\boldsymbol{x}$由仿射变换

$$
\boldsymbol{x} = \boldsymbol{W\boldsymbol{u} + \boldsymbol{b}}
$$


得到。考虑一个由$m$个样本组成的小批量，仿射变换的输出为一个新的小批量$\mathcal{B} = \{\boldsymbol{x}^{(1)}, \ldots, \boldsymbol{x}^{(m)} \}$。它们正是批量归一化层的输入。对于小批量$\mathcal{B}$中任意样本$\boldsymbol{x}^{(i)} \in \mathbb{R}^d, 1 \leq  i \leq m$，批量归一化层的输出同样是$d$维向量

$$
\begin{equation}
\begin{aligned}
\boldsymbol{y}^{(i)}  & = \phi(\text{BN}(\boldsymbol{x}^{(i)}))
\end{aligned}
\end{equation}
$$


并由以下几步求得。首先，对小批量$\mathcal{B}$求均值和方差：

$$
\boldsymbol{\mu}_\mathcal{B} \leftarrow \frac{1}{m}\sum_{i = 1}^{m} \boldsymbol{x}^{(i)}
$$

$$
\boldsymbol{\sigma}_\mathcal{B}^2 \leftarrow \frac{1}{m} \sum_{i=1}^{m}(\boldsymbol{x}^{(i)} - \boldsymbol{\mu}_\mathcal{B})^2
$$


其中的平方计算是按元素求平方。接下来，使用按元素开方和按元素除法对$\boldsymbol{x}^{(i)}$标准化：

$$
\hat{\boldsymbol{x}}^{(i)} \leftarrow \frac{\boldsymbol{x}^{(i)} - \boldsymbol{\mu}_\mathcal{B}}{\sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}},
$$


这里$\epsilon > 0$是一个很小的常数，保证分母大于0。在上面标准化的基础上，批量归一化层引入了两个可以学习的模型参数，**拉伸**（scale）参数 $\boldsymbol{\gamma}$ 和**偏移**（shift）参数 $\boldsymbol{\beta}$。这两个参数和$\boldsymbol{x}^{(i)}$形状相同，皆为$d$维向量。它们与$\boldsymbol{x}^{(i)}$分别做按元素乘法（符号$\odot$）和加法计算：

$$
{\boldsymbol{y}}^{(i)} \leftarrow \boldsymbol{\gamma} \odot \hat{\boldsymbol{x}}^{(i)} + \boldsymbol{\beta}.
$$

至此，我们得到了$\boldsymbol{x}^{(i)}$的批量归一化的输出$\boldsymbol{y}^{(i)}$。
值得注意的是，可学习的拉伸和偏移参数保留了不对$\hat{\boldsymbol{x}}^{(i)}$做批量归一化的可能：此时只需学出$\boldsymbol{\gamma} = \sqrt{\boldsymbol{\sigma}_\mathcal{B}^2 + \epsilon}$和$\boldsymbol{\beta} = \boldsymbol{\mu}_\mathcal{B}$。我们可以对此这样理解：如果批量归一化无益，理论上，学出的模型可以不使用批量归一化。

- 对卷积层做批量归一化

对卷积层来说，批量归一化发生在卷积计算之后、应用激活函数之前。如果卷积计算输出多个通道，我们需要对这些通道的输出分别做批量归一化，且**每个通道都拥有独立的拉伸和偏移参数，并均为标量**。设小批量中有$m$个样本。在单个通道上，假设卷积计算输出的高和宽分别为$p$和$q$。我们需要对该通道中$m \times p \times q$个元素同时做批量归一化。对这些元素做标准化计算时，我们使用相同的均值和方差，即该通道中$m \times p \times q$个元素的均值和方差。

- 预测时的批量归一化

使用批量归一化训练时，我们可以将批量大小设得大一点，从而使批量内样本的均值和方差的计算都较为准确。将训练好的模型用于预测时，我们希望模型对于任意输入都有确定的输出。因此，单个样本的输出不应取决于批量归一化所需要的随机小批量中的均值和方差。一种常用的方法是通过移动平均估算整个训练数据集的样本均值和方差，并在预测时使用它们得到确定的输出。可见，和丢弃层一样，批量归一化层在训练模式和预测模式下的计算结果也是不一样的。

##### 代码实战

- 使用批量归一化方法优化lenet，[->](classical_cnn_models/BN/BatchNormalization.py)
- 核心代码

```python

# 定义一次batch normalization运算的计算图
def batch_norm(is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 判断当前模式是训练模式还是预测模式
    if not is_training:
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算每个通道维上（axis=1）的均值和方差。
            # 这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        # 训练模式下用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 拉伸和偏移
    return Y, moving_mean, moving_var


# 手动实现版本BatchNormalization层的完整定义
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)  # 全连接层输出神经元
        else:
            shape = (1, num_features, 1, 1)  # 通道数
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 不参与求梯度和迭代的变量，全在内存上初始化成0
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var, Module实例的traning属性默认为true, 调用.eval()后设成false
        Y, self.moving_mean, self.moving_var = batch_norm(self.training,
                                                          X, self.gamma, self.beta, self.moving_mean,
                                                          self.moving_var, eps=1e-5, momentum=0.9)
        return Y
```



#### 残差网络（ResNet）

 ResNets要解决的是深度神经网络的“退化”问题。我们知道，对浅层网络逐渐叠加layers，模型在训练集和测试集上的性能会变好，因为模型复杂度更高了，表达能力更强了，可以对潜在的映射关系拟合得更好。而“退化”指的是，给网络叠加更多的层后，性能却快速下降的情况，如图：

<div align=center>
<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.2_introduction_of_image_classification/640.png"/>
</div>

针对这一问题，何恺明等人提出了残差网络（ResNet）。它在2015年的ImageNet图像识别挑战赛夺魁，并深刻影响了后来的深度神经网络的设计。



##### 网络结构

-  残差块

设输入为$\boldsymbol{x}$。假设我们希望学出的理想映射为$f(\boldsymbol{x})$，从而作为图5.9上方激活函数的输入。左图虚线框中的部分需要直接拟合出该映射$f(\boldsymbol{x})$，而右图虚线框中的部分则需要拟合出有关恒等映射的残差映射$f(\boldsymbol{x})-\boldsymbol{x}$。残差映射在实际中往往更容易优化。以本节开头提到的恒等映射作为我们希望学出的理想映射$f(\boldsymbol{x})$。我们只需将图5.9中右图虚线框内上方的加权运算（如仿射）的权重和偏差参数学成0，那么$f(\boldsymbol{x})$即为恒等映射。实际中，当理想映射$f(\boldsymbol{x})$极接近于恒等映射时，残差映射也易于捕捉恒等映射的细微波动。图5.9右图也是ResNet的基础块，即残差块（residual block）。在残差块中，输入可通过跨层的数据线路更快地向前传播。

<div align=center>
<img width="400" src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter02/2.2_introduction_of_image_classification/5.11_residual-block.svg"/>
</div>
<div align=center>普通的网络结构（左）与加入残差连接的网络结构（右）</div>

##### 代码实战

- 可运行代码[->](classical_cnn_models/ResNet/ResNet.py)
- 核心代码

```python
# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
# ResNet
class Net(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(Net, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
```



## 参考文献



[^1]:https://cs231n.github.io/convolutional-networks/#overview CS231   
[^2]:https://poloclub.github.io/cnn-explainer/ CNN Explainer    
[^3]: https://zhuanlan.zhihu.com/p/41423739 ,Amusi, 《一文读懂VGG网络》    
[^4]: https://zhuanlan.zhihu.com/p/32702031 ，张磊，深入理解GoogLeNet结构





**Task03 CNN基础**

--- ***By: QiangZiBro***


>https://github.com/QiangZiBro


**关于Datawhale**：

>Datawhale是一个专注于数据科学与AI领域的开源组织，汇集了众多领域院校和知名企业的优秀学习者，聚合了一群有开源精神和探索精神的团队成员。Datawhale以“for the learner，和学习者一起成长”为愿景，鼓励真实地展现自我、开放包容、互信互助、敢于试错和勇于担当。同时Datawhale 用开源的理念去探索开源内容、开源学习和开源方案，赋能人才培养，助力人才成长，建立起人与人，人与知识，人与企业和人与未来的联结。


