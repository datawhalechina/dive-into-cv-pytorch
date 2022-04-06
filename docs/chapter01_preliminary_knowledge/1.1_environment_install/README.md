# GPU环境配置

进行CV的学习，配置好实验环境就是第一步。本小节就带大家简单的过一下如何配置深度学习GPU环境。

主要分为4个部分

- 如何获取GPU资源

- CUDA安装

- Cudnn安装

- Anaconda安装及环境管理

- Pytorch安装

## 1.如何获取GPU资源？

想要更高效的进行CV任务的训练，你的电脑就必须要有一个得力干将，那就是GPU，而且需要是NVIDIA的显卡，因为我们需要CUDA。

如果你的电脑拥有GPU，那么你可以直接跳过这一小节。

如果你没有GPU资源，那么有几条路可以选：

- 薅一些GPU资源云服务提供商的羊毛

- 选择短期租用GPU服务器

- 自己购买GPU改造台式机

### 1.1 哪里可以薅羊毛？

这里简单罗列几个提供免费算力的GPU计算资源平台, 这样你就可以愉快的做(薅)实(羊)验(毛)了～

`Kaggle Kernel`

Kaggle竞赛平台在新建一个Kernel时，是可以通过设置中选择GPU，每周可以免费使用30小时。

`Google Colab`

和Kaggle的Kernel类似，Goggle的Colab本质上是一个在线版本的jupyter notebook，可以免费嫖一个NVIDIA Tesla K80用。

`天池实验室`

[天池Notebook](https://tianchi.aliyun.com/notebook-ai/)，也是一个很好的选择，提供免费GPU算力进行在线编程实验，上面还可以看到其他人分享的代码。

`AI Studio 与 PaddlePaddle`

百度在推进深度学习发展上也做了不少努力，目前提供的免费算力还是很感人的，感兴趣也可以尝试。


### 1.2 如何给电脑装GPU?

实际上由于游戏行业的普及，目前很多人的电脑，不论是台式机还是笔记本，都是有独显的，也就是GPU。

现在的gpu普遍价格不菲，且更新换代较快，因此如果是个人使用且资金不是特别充裕，还是更建议短期租用gpu服务器的方式进行使用，而不是购买gpu。但不管怎样，如果你的电脑没有一块NVIDIA GPU，你可以选择买一块自己装！

实际上，自己买一块GPU并自己安装，并没有什么过多的技术含量，当然也有几个要注意的要点：

- 你购买的GPU要和你的电脑其他配件的实力相匹配

这个其实不是什么硬性要求，但是GPU和主板及CPU能力的大致匹配是最好的，为什么呢？

如果你的GPU是3080Ti，而你的cpu性能不够，那你在训练中很可能遇到cpu已经满负荷运转，而GPU的利用率还很低的情况，相当于浪费了一定的GPU计算能力。

- 关注GPU的功率，更换电源

如果你买的GPU功耗比较高，为了GPU能够长期正常安全使用，你需要查一下你买的GPU的功率，然后配一个相应的电源。通常来说，你可以这样来操作，举个例子：

假如你现在的电脑没有一个好的独显，然后当前电源的功率是400w。你想买一个GTX1070，那么可以查到，1070公版显卡的标准功耗是150W，所以比较推荐配一个600W的电源，留出这部分增量。

- 如何接线

实际上，更换电源及GPU都是非常简单的，各种接线口的设计都已经考虑了你不懂的因素，基本可以这么说，只要形状对应，你能轻松插上去，就是接对了。

当然，各种接口都是“防呆不防傻”的，插不进去别楞来，可以搜索相关的装机指南辅助进行安装。

## 2.如何安装CUDA?

有了GPU机器，下面要做的就是安装CUDA。如果你是租用或者蹭的云服务器，通常来说CUDA已经帮你完成，可以直接跳过本节。

由于NVIDIA GPU及其配套驱动以及CUDA的版本的不断更新，网上能搜到众多不同系统下不同CUDA版本的安装教程，你一定适用于你的情况，强烈建议遵循NVIDIA官网教程进行安装。

这里简单帮你梳理下Ubuntu18.04下的安装方法：

首先是CUDA版本的选择，你能安装的最高CUDA版本，取决于你的GPU硬件型号。

目前来说，CUDA10和CUDA11是比较广泛使用的版本，再之前的版本就有些过于老旧了，不建议安装。（本段修改于2022/04/05）

置于安装，比较推荐的方法分为两步：安装NVIDIA驱动 + 安装CUDA

你的电脑在系统装好后就已经自带了显卡驱动，但是我们需要安装NIVIDA的驱动。

在Ubuntu18.04下，NVIDIA驱动的安装已经变得非常简单，只需一行命令

`$ sudo ubuntu-drivers autoinstall`

如有疑惑详见[How to install the NVIDIA drivers on Ubuntu 18.04 Bionic Beaver Linux](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-18-04-bionic-beaver-linux#h7-automatic-install-using-ppa-repository-to-install-nvidia-beta-drivers)

接下来，到NVIDIA官网下载CUDA的安装文件,[这个页面](https://developer.nvidia.com/cuda-toolkit-archive)可以找到不同版本的下载链接，这里我们以CUDA10.0为例


<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter01/1.1_environment_install/select_CUDA_version.png">

进入CUDA10.0的下载界面，根据自己的系统进行勾选，并下载CUDA安装文件

<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter01/1.1_environment_install/select_installer.png">

最后就是安装了，运行如下命令：

`$ sudo sh cuda_10.0.130_410.48_linux.run`

安装过程比较简单，核心要注意两点：

- 安装过程会询问你是否安装驱动。这里一定要选择否，因为我们前面已经手动安装过了。
- 选择cuda的安装路径

不同版本的安装过程会稍有不同，但基本方法一致，都是把握好上述两点即可。

下面展示一下另一个版本CUDA11.1的安装过程截图 (此处更新于2022/4/5)

运行安装后，会看到如下界面，注意Driver要取消勾选，CUDA ToolKit一定要选，其他如Samples是一些CUDA示例代码，可以取消勾选不装。


<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter01/1.1_environment_install/cuda11_install_1.png">

然后进入Options设置其他相关参数，一个比较重要的参数是 Options -> CUDA Toolkit -> Change Toolkit Install Path，这个参数用来指定CUDA安装位置

<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter01/1.1_environment_install/cuda11_install_2.png">

<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter01/1.1_environment_install/cuda11_install_3.png">

最后，执行Install，等待一会即可完成安装。如下是没有选择安装驱动时安装成功的运行结果示例：

<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter01/1.1_environment_install/cuda11_install_4.png">

安装完成后设置环境变量，在~/.bashrc文件结尾添加如下两句(注意将路径修改为你自己的cuda安装路径)

`export PATH=/usr/local/cuda-10.0/bin:$PATH`

`export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64`

然后重新加载bashrc文件

`$ source ~/.bashrc`

到此，如果顺利的话CUDA已经安装成功了，你可以使用如下命令验证下：

```
$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Sat_Aug_25_21:08:01_CDT_2018
Cuda compilation tools, release 10.0, V10.0.130
```

这个过程应该是整个环境配置过程中最容易出问题的，一般都和显卡驱动有关。不要灰心，谁没在安装CUDA上面浪费过几天时间呢～

## 3.如何安装Cudnn

首先到[官网下载cudnn](https://developer.nvidia.com/cudnn)，注意版本一定要和CUDA进行对应就好

cudnn的安装，其实就是将文件对应的拷贝到你安装的cuda相应目录下。

下载并解压后，参照下面命令执行即可（注意替换为你自己的路径）
```
$ sudo cp cuda/include/cudnn.h /usr/local/cuda/include
$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
$ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

## 4.Anaconda安装及环境管理

Anaconda是目前非常流行的一个python包管理器，自带很多流行的python库，包括numpy，pandas等，当然还有conda。而Conda是一个开源的软件包管理系统和环境管理系统，用于安装多个版本的软件包及其依赖关系，并在它们之间轻松切换。我们非常推荐使用Anaconda来管理你的python环境，随着你后面的不断学习，你会感受到它的精髓和好用。

这里我们首先去下载Anaconda的“缩水版”[Minconda](https://docs.conda.io/en/latest/miniconda.html)

运行下载的文件即可完成miniconda的安装

`$ bash miniconda相应版本.sh`

安装过程中会询问你，是否需要帮你将软件添加的环境变量中，你可以选择是，也可以自己手动添加，就像之前给CUDA添加环境变量那样。

安装非常简单，不会有什么问题，下面我们来介绍基于conda的环境管理。

**创建一个虚拟环境**

使用如下命令创建一个名为py37_torch131的指定python3.7版本的新环境。

`$ conda create -n py37_torch131 python=3.7`

**查看所有环境**

使用`conda env list`命令可以查看当前的所有python环境

```
$ conda env list
# conda environments:
#
base                  *  /home/ansheng/miniconda3
open-mmlab               /home/ansheng/miniconda3/envs/open-mmlab
py37_torch131            /home/ansheng/miniconda3/envs/py37_torch131
```

可以看到，我的电脑上有3个环境。一个是安装anaconda后自带的base环境，一个是我们刚刚新建的python环境py37_torch131，还有一个是我之前创建的环境。

通过conda可以很方便的管理多个不同的环境，从而避免可能出现的版本问题。通常我们可以为pytorch建一个环境，为tensorflow建一个环境。

**删除一个虚拟环境**

可以通过如下命令删除制定虚拟环境，当然这里我们就不真的实际运行了

`$ conda env remove -n py37_torch131`

**激活or关闭指定环境**

使用`source activate 环境名`来激活指定环境

`$ source activate py37_torch131`

使用`source deactivate` or `source deactivate 环境名` 关闭当前环境

`$ source deactivate py37_torch131`

**在指定环境中安装包**

下面我们介绍如何在指定环境中安装需要的库，同样非常简单。

首先重新激活刚刚创建的环境

`$ source activate py37_torch131`

使用 `pip install package_name` 命令安装相应包

例如，下面命令安装进度条辅助工具库 tqdm

`pip install tqdm`

下面是一些常用工具库，你可以使用pip命令一并安装到当前环境中：

`$ pip install jupyter tqdm opencv-python matplotlib numpy pandas`

**下载超时的解决办法**

使用pip或conda时遇到下载龟速甚至超时失败的情况，可以通过更好国内源的方式解决

pip更换清华源

```
$ mkdir ~/.pip
$ cd ~/.pip
$ vi pip.conf
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

conda更换清华的源

```
$ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
$ conda config --set show_channel_urls yes
```

## 5.pytorch安装

本教程的所有实战代码都是基于Pytorch这个深度学习框架的，因此我们要先将pytorch安装好，来完成后面的学习。

不同环境下的安装命令可以在[Pytorch官网](https://pytorch.org)找到(你可以结合gpu型号选择适合自己的版本)

<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter01/1.1_environment_install/select_pytorch.png">

例如：使用如下命令安装gpu_1.3.1版本的pytorch

`$ conda install pytorch=1.3.1 torchvision cudatoolkit=10.0`

上面的命令相当于通过conda帮你在当前环境中把pytorch及其所有的依赖库都安装好了。

安装pytorch有一个比较容易出现的问题，就是错误安装了cpu版本，使得gpu无法使用

在python中，使用如下命令查看pytorch版本以及gpu是否可用

<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter01/1.1_environment_install/torch_test.png">

其中，`torch.cuda.is_available()` 返回 True，说明安装正确


---

到此，本教材关于GPU环境配置的介绍就全部结束了。如有疑问和建议，欢迎交流。

**贡献者**

[安晟](https://github.com/monkeyDemon)

