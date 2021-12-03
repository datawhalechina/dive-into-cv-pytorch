# 前言

## 本章内容简介

[Generative Adversarial Network](https://arxiv.org/pdf/1406.2661.pdf)，就是大家耳熟能详的GAN，由Ian Goodfellow首先提出，是近年来计算机视觉领域相对热门的研究方向之一。

实际上，GAN技术的各种应用已经悄悄走进了我们的生活中。数据科学家和深度学习研究者使用这项技术来进行各种有趣而又富有创造性的实验，例如**生成艺术作品**，**改变面部表情**，**创建游戏场景（3D重建）**，**可视化设计**等等。而之前较火的恶搞类视频肌肉金轮也是借助GAN技术实现AI换脸。

<img src="https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter05/GAN的应用.png" alt="GAN的应用" style="zoom:50%;" />

希望通过本章的学习，可以帮助大家了解GAN模型的基本原理和训练方法。本章将大致分为以下几个小节来带领大家走进GAN的世界：

- 5.1 初识生成对抗网络
- 5.2 GAN实战: 手写数字生成
- 5.3 ConditionGAN实战: 再战手写数字生成
- 5.4 DCGAN实战: 深度卷积生成对抗网络

## 内容设计

本章节的代码和教程均由[沈豪](https://github.com/shenhao-stu)完成，并最终由安晟进行修改和校订。

本章节设计的理论部分参考自[邱锡鹏 神经网络与深度学习](https://nndl.github.io/)

## 教程定位

1. 定位人群：对CNN和Pytorch有一定了解，至少跑过分类网络的训练，有一点数学基础
2. 时间安排：7天，每天花费1-3小时，视个人基础而定
3. 学习类型：理论教程+代码实践
4. 难度系数：🌟🌟🌟

