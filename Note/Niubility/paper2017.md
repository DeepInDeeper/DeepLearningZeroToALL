### 盘点2017深度学习论文
[原文链接](https://kloudstrifeblog.wordpress.com/2017/12/15/my-papers-of-the-year/)  

---
 体系结构/模型
* [SMASH : one shot model architecture search through Hypernetworks](https://arxiv.org/pdf/1708.05344.pdf)
今年关于网络架构的文章已经少得多了，好多东西都或多或少的稳定了。尽管有些论文肯定还是想要推动这个框架发展。首当其冲的是安德鲁·布鲁克（Andrew Brock）的破解SMASH网络结构，它是在1000个GPU上来进行神经架构的搜索。

* [Densely connected convolutional networks](https://arxiv.org/pdf/1608.06993.pdf)
DenseNets（2017年更新的修订版本）是一个对内存需求很大(memory-hungry)，但非常干净利落的想法。 简单的说就是“在计算机视觉中，眼睛+毛皮=猫，所以想法就是连接所有的东西（和连接层）”。

* [Scaling the Scattering Transform](https://arxiv.org/pdf/1703.08961.pdf)
在CNNs中有一个非常重要但是被低估的思想是散射变换，有效地取一个小波滤波器组（将小波理论与conv + maxpool和ReLU相连）的模。不知何故，这揭示了为什么头几层看起来像Gabor滤波器，并且为什么你可能根本不需要训练它们。用Stephane Mallat的话来说，“我很惊讶它有效”。 见下面的论文。

* [Tensorized LSTMs for sequence learning](https://arxiv.org/pdf/1711.01577.pdf)
张量化的LSTM是维基百科上新的SotA编码，每个字符占用1.2位。有些人认为英文的编码限制是1.0,1.1 BPC（仅供参考，LayerNorm LSTM大约为1.3 bpc）。由于想法新颖的原因，我更喜欢这篇论文“Recurrent Highway HyperNetworks”。

最后，不需要进一步评论的论文：  
* [Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf)

* [Matrix capsules with EM routing](https://openreview.net/pdf?id=HJWLfGWRb)

---
生成模型

由于NVidia公司爆炸性的原因，我有意将大部分GAN论文放在Progressive Growing这部分论文中。
* [Neural Discrete Representation Learning](https://arxiv.org/pdf/1711.00937.pdf)
首先是与自我回归内容相关的家族--Aaron Van den Oord的最新杰作VQ-VAE，是事后看起来很显然的那种论文之一，提出的双停止梯度损失函数肯定是经过非常多的努力。我相信大量的迭代 - 包括用ELBO的贝叶斯层ala PixelVAE来做中间层 - 将会从这项工作中衍生出来。

* [Parallel Wavenet](https://arxiv.org/pdf/1711.10433.pdf)
另外一个惊喜是来自Parallel Wavenet的结构。当大家都期待与Tom LePaine的工作一起的一个快速二进制结构，DeepMind公司的家伙给了我们一个师生蒸馏模型和噪音整形，其中噪音整形是将高维各向同性的高斯/逻辑潜在空间解释为可以通过逆自回归流进行自回归的时间过程。非常非常干净利落的想法。


* [Progressive growing of GANs](https://arxiv.org/pdf/1710.10196.pdf)
没人预料到的头号文件 – Nvidia公司制定了法律。GAN理论是完全循环的 - 而不是Wassersteinizing（借用Justin Solomon的不朽之语），只是保持KL损失，但是通过对数据分布进行多分辨逼近来消除不相交的支持问题。这仍然需要一些技巧来稳定梯度，但经验结果不言自明。

* [The VeGAN cookbook](https://arxiv.org/pdf/1705.07642.pdf)
* [Wasserstein Autoencoders](https://arxiv.org/pdf/1711.01558.pdf)
尽管由Peyre和Genevay领导的法国研究小组在今年早些时候确定了最小Kantorovich估计量，但由Bousquet领导的Google团队在前进/最佳运输设置中写下了VAE-GAN的决定性框架。W-AAE的论文可能会成为ICLR2018的热门话题之一。

* [Hierarchical Implicit Models](https://arxiv.org/pdf/1702.08896.pdf)
在变分推理方面，谁比德斯廷·特朗更好地借鉴了非政策强化学习和GANs的观点，并再次推动了现代VI能做的事情：
---
强化学习

被轻(soft)/最大熵Q学习占主的一年。但是这些年来！我们一直做错了！
*  [Equivalence between Policy Gradients and Soft Q-learning](https://arxiv.org/pdf/1704.06440.pdf)
Nuff说，在一篇里程碑式的论文里：舒尔曼证明了RL算法主要两个族的等价性。

* [Bridging the gap between value and policy RL](https://arxiv.org/pdf/1702.08892.pdf)
他是否会做到这一点，通过采取古老的数学和非常仔细的重新做分区函数的计算，来证明路径到路径等价？ 没有人知道，除了Ofir：

* [A unified view of entropy-regularized MDPs](https://arxiv.org/pdf/1705.07798.pdf)
另外一篇被低估的论文 - Gergely悄悄地把所有的RL都算在上面的RL算法和凸优化方法之间。对于今年论文的恕我直言，RL相关的论文是一个有力的竞争者，但很少有人听说过。

* [Imagination-Augmented Agents](https://arxiv.org/pdf/1707.06203.pdf)
如果David Silver的Predictron由于在ICLR 2017被拒绝（！）而以某种方式掉下了雷达，那么Theo的论文就像是一个双重的观点，它以优美而直观的Sokoban实验结果来启动了它：

* [A distributional perspective on RL](https://arxiv.org/pdf/1707.06887.pdf)
* [Distributional RL with Quantile Regression](https://arxiv.org/pdf/1710.10044.pdf)
马克·贝莱马尔（Marc Bellemare）获得了另外一个转型的论文 - 废除了所有的DQN稳定插件，并简单地学习了分发（并且在这个过程中击败了SotA）。非常漂亮的论文。有许多可能的扩展，包括与Wasserstein距离的链接。

* [Noisy Networks for Exploration](https://arxiv.org/pdf/1706.10295.pdf)
一个简单，但非常有效，双重whammy的想法。


* [Mastering the game of Go without human knowledge](https://deepmind.com/documents/119/agz_unformatted_nature.pdf)
当然，如果没有AlphaGo Zero的话，这个列表还是不完整的。将策略网络MCTS前后对齐的思想，例如将MCTS作为一种策略改进算法（以及采用使NN近似误差平滑而不是传播的方法）是一种神话般的东西。



SGD和优化

对于为什么SGD算法在非凸面情况下的也能效果好的原理探索（从广义误差角度来看如此难以打败），2017年的工作已经让它变得成熟了。

* [Deep Relaxation : PDEs for optimizing deep networks](https://arxiv.org/pdf/1704.04932.pdf)
今年的“最技术”论文是Chaudhari。从SGD和梯度流向PDE几乎连接了一切。遵循并完成“Entropy-SGD”原则的一篇杰作：

* [SGD as approximate Bayesian inference](https://arxiv.org/pdf/1704.04289.pdf)
贝叶斯的观点认为这是Mandt＆Hoffman的SGD-VI连接。如你所知，我多年来一直是一个频繁的人，原文如此。

* [Batch size matters, a diffusion approximation framework](https://arxiv.org/pdf/1705.07562.pdf)
前面的文章取决于SGD作为随机微分方程的连续松弛（由于CLT，梯度噪声被视为高斯）。 这解释了批量大小的影响，并给出了一个非常好的卡方公式。

* [Three factors influencing minima in SGD](https://arxiv.org/pdf/1711.04623.pdf)
另一篇与Ornstein-Uhlenbeck类似的论文启发了Yoshua Bengio实验室的研究成果：


* [SGD performs VI, converges to limit cycles](https://arxiv.org/pdf/1710.11029.pdf)
最后， Chandhari先生又一次对SGD-SDE-VI三位一体的贡献：

---
理论

* [Opening the black box of deep networks via information / On the information bottleneck theory of deep learning](https://arxiv.org/pdf/1703.00810.pdf)
我坚信一个直觉，就是为什么深度学习的工作将来自谐波/二阶分析（前面已经看到散射思想）和信息理论与基于熵的测量之间的交集。Naftali Tishby的想法虽然仍然有争议，但最近ICLR2018提交的内容，使我们更加接近这一理解。


* [Deep variational information bottleneck](https://arxiv.org/pdf/1612.00410.pdf)
同样，来自ICLR2017的一篇漂亮的论文对信息瓶颈理论采取了一种变化的方法。我的选择与“解开”类别中的Beta-VAE相比有一点小小的优势。

* [A Lagrangian perspective on latent variable modelling](https://openreview.net/pdf?id=ryZERzWCZ)
今年已有数十亿个生成模型，其中有12亿个因子分解对数似然的方法，他们大都可以被理解为凸二元。恕我直言，一篇非常必要的论文。


* [Geometry of NN loss surfaces via RMT / Nonlinear RMT for deep learning](http://proceedings.mlr.press/v70/pennington17a/pennington17a.pdf)
最后，杰夫·潘宁顿(Jeff Pennington )以惊人的技术手段显示了深度学习中关于数学的军备竞赛仍然活跃，他将复杂的分析，随机矩阵理论，自由概率和图形态射（！）结合起来，导出了神经网络损失函数的Hessian特征值的确切定律，而其图形只是靠经验才先知道的，例如在Sagun等人的论文中有显示。必读。

