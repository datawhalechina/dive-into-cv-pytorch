# CV 大作业：Fashion-mnist分类任务

经典的MNIST数据集包含了大量的手写数字。十几年来，来自机器学习、机器视觉、人工智能、深度学习领域的研究员们把这个数据集作为衡量算法的基准之一。你会在很多的会议，期刊的论文中发现这个数据集的身影。实际上，MNIST数据集已经成为算法作者的必测的数据集之一。有人曾调侃道："如果一个算法在MNIST不work, 那么它就根本没法用；而如果它在MNIST上work, 它在其他数据上也可能不work！"

Fashion-MNIST的目的是要成为MNIST数据集的一个直接替代品。作为算法作者，你不需要修改任何的代码，就可以直接使用这个数据集。Fashion-MNIST的图片大小，训练、测试样本数及类别数与经典MNIST完全相同。

本次任务需要针对Fashion-MNIST数据集，设计、搭建、训练机器学习模型，能够尽可能准确地分辨出测试数据地标签。

评价指标：本次任务采用 ACC（Accuracy) 作为模型的评价标准。

最后提交一个csv文件，格式如下：

|  ID   | Prediction  |
|  ----  | ----  |
| 0  | 4(预测类别) |
| 1  | 9 |
| 2  | 3 |

## baseline 0.9235

为方便大家学习，这里给出一个比较基本的baseline，得分0.9235

主要是给没有基础的小伙伴一个指引，包括如何下载数据集，保存模型，生成提交结果的一个简单流程。

详见目录：`baseline`

## 2020/2/22更新 0.9432:rocket:

天又做了一些简单改动，将分数提到了0.9432

主要是3点改动：输入归一化、数据增强、sgd

详见目录：`baseline_plus`

## 2020/2/26更新 0.9526:rocket:

仅仅在之前的基础上，添加学习率阶段性下降的策略，即可将分数进一步提升到0.9526

详见目录：`baseline_plus`

## 未完待续...

即将更新 0.9625 分解决方案, 敬请期待:beer:

## 参考文献

[1] Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. Han Xiao, Kashif Rasul, Roland Vollgraf. arXiv:1708.07747

[2] https://github.com/zalandoresearch/fashion-mnist/
