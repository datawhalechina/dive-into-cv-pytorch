
在下一小节中，我们将对最初版的GAN进行模型分析，了解一开始的GAN网络存在着哪些问题。

## 模型分析

还记得之前提到的将整个生成对抗网络的目标函数看作是**最小化最大化游戏（Minimax Game）**。
$$
\begin{aligned}
& \min _{\theta} \max _{\phi}\left(\mathbb{E}_{\boldsymbol{x} \sim p_{r}(x)}[\log D(\boldsymbol{x} ; \phi)]+\mathbb{E}_{\boldsymbol{x} \sim p_{\theta}(x)}[\log (1-D(\boldsymbol{x} ; \phi))]\right) \\
=& \min _{\theta} \max _{\phi}\left(\mathbb{E}_{\boldsymbol{x} \sim p_{r}(x)}[\log D(\boldsymbol{x} ; \phi)]+\mathbb{E}_{z \sim p(z)}[\log (1-D(G(\boldsymbol{z} ; \theta) ; \phi))]\right)
\end{aligned} \tag{6}
$$
由于的生成网络梯度问题，这个最小化最大化形式的目标函数一般用来进行理论分析，并不是实际训练时的目标函数。

对于判别器模型，它的min**损失函数**为：
$$
\mathcal{L}(f)=\mathbb{E}_{\boldsymbol{x} \sim p\left(\boldsymbol{x} \mid c_{1}\right)}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{x} \sim p\left(\boldsymbol{x} \mid c_{2}\right)}[\log (1-D(\boldsymbol{x}))] \tag{7}
$$
假设$p_{r}(\boldsymbol{x})$和$p_{\theta}(\boldsymbol{x})$已知，通过数学推导，可以得到最优的判别器为
$$
D^{\star}(\boldsymbol{x})=\frac{p_{r}(\boldsymbol{x})}{p_{r}(\boldsymbol{x})+p_{\theta}(\boldsymbol{x})} \tag{8}
$$

将此时的$D^{\star}(x)$带入损失函数中，其目标函数变为
$$
\begin{aligned}
\mathcal{L}\left(G \mid D^{\star}\right) &=\mathbb{E}_{\boldsymbol{x} \sim p_{r}(x)}\left[\log D^{\star}(\boldsymbol{x})\right]+\mathbb{E}_{\boldsymbol{x} \sim p_{\theta}(x)}\left[\log \left(1-D^{\star}(\boldsymbol{x})\right)\right] \\
&=\mathbb{E}_{\boldsymbol{x} \sim p_{r}(x)}\left[\log \frac{p_{r}(\boldsymbol{x})}{p_{r}(\boldsymbol{x})+p_{\theta}(\boldsymbol{x})}\right]+\mathbb{E}_{\boldsymbol{x} \sim p_{\theta}(x)}\left[\log \frac{p_{\theta}(\boldsymbol{x})}{p_{r}(\boldsymbol{x})+p_{\theta}(\boldsymbol{x})}\right] \\
&=\mathrm{KL}\left(p_{r}, p_{a}\right)+\mathrm{KL}\left(p_{\theta}, p_{a}\right)-2 \log 2 \\
&=2 \mathrm{JS}\left(p_{r}, p_{\theta}\right)-2 \log 2
\end{aligned} \tag{9}
$$

其中$\mathrm{JS}(\cdot)$ 为 $\mathrm{JS}$ 散度, $p_{a}(\boldsymbol{x})=\frac{1}{2}\left(p_{r}(\boldsymbol{x})+p_{\theta}(\boldsymbol{x})\right)$ 为一个“平均”分布。

在生成对抗网络中，当判别网络为最优时，生成网络的优化目标是最小化真实分布$p_r$和模型分布$p_{\theta}$之间的$JS$散度。当两个分布相同时，$JS$散度为0，最优生成网络$G^{\star}$对应的损失为$\mathcal{L}\left(G^{\star} \mid D^{\star}\right)=−2log2$。

### 训练稳定性

使用 $JS$ 散度来训练生成对抗网络的一个问题是当两个分布没有重叠时，它们之间的$JS$散度恒等于常数$log 2$。对生成网络来说，目标函数关于参数的梯度为0，即$\frac{\partial \mathcal{L}\left(G \mid D^{\star}\right)}{\partial \theta}=0$。

当真实分布 $p_r $和模型分布 $p_{\theta} $没有重叠时，最优的判别器$D^{\star}$对所有生成的数据的输出都为0，而从导致生成网络的梯度消失。

因此，在实际训练生成对抗网络时，**一般不会将判别网络训练到最优**，只进行**一步或多步梯度**下降，使得生成网络的梯度依然存在。另外，判别网络也不能太差，否则生成网络的梯度为错误的梯度。但是，如何在梯度消失和梯度错误之间取得平衡并不是一件容易的事，这个问题使得生成对抗网络在训练时稳定性比较差。

![训练稳定性](img/训练稳定性.png)

### 模型坍塌（mode collapse）

对于生成器的另一种奖励形式的目标函数，将$G^{\star}$带入得到：
$$
\max _{\theta}\left(\mathbb{E}_{\boldsymbol{z} \sim p(z)}[\log D(G(\boldsymbol{z} ; \theta) ; \phi)]\right) \tag{10}
$$

$$
\begin{array}{l}
\mathcal{L}^{\prime}\left(G \mid D^{\star}\right)=\mathbb{E}_{\boldsymbol{x} \sim p_{\theta}(x)}\left[\log D^{\star}(\boldsymbol{x})\right] \\
\quad=\mathbb{E}_{\boldsymbol{x} \sim p_{\theta}(x)}\left[\log \frac{p_{r}(\boldsymbol{x})}{p_{r}(\boldsymbol{x})+p_{\theta}(\boldsymbol{x})} \cdot \frac{p_{\theta}(\boldsymbol{x})}{p_{\theta}(\boldsymbol{x})}\right] \\
=-\mathbb{E}_{x \sim p_{\theta}(x)}\left[\log \frac{p_{\theta}(\boldsymbol{x})}{p_{r}(\boldsymbol{x})}\right]+\mathbb{E}_{\boldsymbol{x} \sim p_{\theta}(x)}\left[\log \frac{p_{\theta}(\boldsymbol{x})}{p_{r}(\boldsymbol{x})+p_{\theta}(\boldsymbol{x})}\right] \\
=-\mathrm{KL}\left(p_{\theta}, p_{r}\right)+\mathbb{E}_{\boldsymbol{x} \sim p_{\theta}(x)}\left[\log \left(1-D^{\star}(\boldsymbol{x})\right)\right] \\
=-\mathrm{KL}\left(p_{\theta}, p_{r}\right)+2 \operatorname{JS}\left(p_{r}, p_{\theta}\right)-2 \log 2-\mathbb{E}_{x \sim p_{r}(x)}\left[\log D^{\star}(\boldsymbol{x})\right]
\end{array} \tag{11}
$$

其中后两项和生成网络无关，因此：
$$
\underset{\theta}{\arg \max } \mathcal{L}^{\prime}\left(G \mid D^{\star}\right)=\underset{\theta}{\arg \min } \mathrm{KL}\left(p_{\theta}, p_{r}\right)-2 \mathrm{JS}\left(p_{r}, p_{\theta}\right) \tag{12}
$$
其中$JS$散度$JS(𝑝_𝜃, 𝑝_𝑟) ∈ [0, log 2]$为有界函数，因此生成网络的目标更多的是受逆向KL散度$KL(p_{\theta},p_r)$影响，使得生成网络更倾向于生成一些更“安全”的样本，从而造成模型坍塌（Model Collapse）问题。

下图给出数据真实分布为一个高斯混合分布，模型分布为一个单高斯分布时，使用前向和逆向 KL 散度来进行模型优化的示例。黑色曲线为真实分布$ 𝑝_𝑟$的等高线，红色曲线为模型分布$𝑝_{\theta}$的等高线.

![模型坍塌](img/模型坍塌.png)

- 在前向KL散度会鼓励模型分布$p_{\theta}(𝒙)$尽可能覆盖所有真实分布$p_r(𝒙)>0$的点，而不用回避$p_r(𝒙)≈0$的点；
- 逆向KL散度会鼓励模型分布$p_{\theta}(𝒙)$尽可能避开所有真实分布$p_r(𝒙)≈0$的点，而不需要考虑是否覆盖所有真实分布$p_r(𝒙)>0$的点。

**一个比较直观的演示：**

![模型坍塌案例](img/模型坍塌案例.png)

可以看到在**生成网络**生成的图中，有一种类的图片重复出现了多次，只是变换了头发的颜色，但整体极其相似。这就是模型崩塌的典型的例子。	

在生成对抗网络中，JS 散度不适合衡量生成数据分布和真实数据分布的距离。由于通过优化交叉熵（JS散度）训练生成对抗网络会导致训练稳定性和模型坍塌问题，因此要改进生成对抗网络，就需要改变其损失函数。比如**W-GAN**用***Wasserstein***距离替代 JS 散度来优化训练的生成对抗网络等等。

### 小结

在本节内容中，我们主要了解到了由于损失函数实际上是用JS散度真实分布和模型分布之间的距离。但是当两个分布没有重叠时，JS散度会等于一个常数，从而导致梯度无法更新，训练稳定性无法保证。同时 ，又由于生成网络的目标更多的是受逆向KL散度影响，从而会造成模型坍塌问题。

因此我们可以看到初始的GAN网络模型依旧存在较多的问题，但却使得深度学习模型具有了创新型，而这也是深度学习为什么影响人们的一点。最近微软小冰中的虚拟形象，相信读者们也印象深刻，彷佛是一张张真实存在于我们身边的人脸，而其背后的原理，也是从GAN网络不断衍生发展而来的。
