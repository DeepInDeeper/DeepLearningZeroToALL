更新：   
* [深度学习2017优秀论文](../Note/Niubility/paper2017.md) 
## 2.2 优化      
**[X]**Ruder S. "**An overview of gradient descent optimization algorithms**"
[[pdf]](https://arxiv.org/abs/1609.04747)**(梯度下降优化算法综述，2017，
[中文翻译](http://blog.csdn.net/heyongluoyao8/article/details/52478715))
** :star::star::star:    
  
深度学习路线图框架  
1 深度学习历史和基础  
1.0 书籍  
1.1 调查  
1.2 深度信念网络（DBN）（深度学习里程碑）  
1.3 ImageNet 演变（深度学习突破点）
1.4 语音识别的演变   
2 深度学习  
2.1 模型   
2.2 优化  
2.3 无监督学习/深度生成模型   
2.4 循环神经网络 / 序列模型  
2.5 神经图灵机  
2.6 深度强化学习  
2.7 深度迁移学习 /终身学习 /  
2.8 一次学习（深度学习） 
3 应用  
3.1 自然语言处理（NLP） 
3.2 物体检测  
3.3 视觉追踪  
3.4 看图说话  
3.5 机器翻译  
3.6 机器人技术  
3.7 艺术方面  
3.8 对象分割   
  

原文出处：https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap  
同样，你可以在[原文](https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap)使用脚本下载一下论文  

#  深度学习论文阅读路线图
> 如果你是深度学习领域的新手，你可能会遇到的第一个问题是：“我应该从哪些文件开始阅读？
> 这里给出深度学习论文阅读的路线图！

路线图按照以下四个指导原则：
- 从大纲到细节
- 从原始到现代
- 从泛型到特定
- 集中最先进的（state-of-the-art）
你会发现很多很新的文章，但真的值得一读.  
我也将继续在路线图继续添加文章


=======

# 1 深度学习历史和基础   
## 1.0 书籍    
**[0]** Bengio, Yoshua, Ian J. Goodfellow, and Aaron Courville. "**Deep learning**." An MIT Press book. (2015). [[html]](http://www.deeplearningbook.org/) **(深度学习圣经，你可以在阅读以下论文时阅读这本书。)** :star::star::star::star::star:    
>>>>>>> 0232d0c39e7818e9c953bea2446dc0cac39e4e39

## 1.1 调查
**[1]** LeCun, Yann, Yoshua Bengio, and Geoffrey Hinton. "**Deep learning**." Nature 521.7553 (2015): 436-444. [[pdf]](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf) **(三巨头的调查)** :star::star::star::star::star:  

## 1.2 深度信念网络（DBN）（深度学习里程碑）  
**[2]** Hinton, Geoffrey E., Simon Osindero, and Yee-Whye Teh. "**A fast learning algorithm for deep belief nets**." Neural computation 18.7 (2006): 1527-1554. [[pdf]](http://www.cs.toronto.edu/~hinton/absps/ncfast.pdf)**(深度学习前夕)** :star::star::star:  

**[3]** Hinton, Geoffrey E., and Ruslan R. Salakhutdinov. "**Reducing the dimensionality of data with neural networks**." Science 313.5786 (2006): 504-507. [[pdf]](http://www.cs.toronto.edu/~hinton/science.pdf) **(里程碑，展示深度学习)** :star::star::star:  

## 1.3 ImageNet 演变（深度学习突破点） 
**[4]** Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "**Imagenet classification with deep convolutional neural networks**." Advances in neural information processing systems. 2012. [[pdf]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) **(AlexNet, 深度学习的突破)** :star::star::star::star::star:  

**[5]** Simonyan, Karen, and Andrew Zisserman. "**Very deep convolutional networks for large-scale image recognition**." arXiv preprint arXiv:1409.1556 (2014). [[pdf]](https://arxiv.org/pdf/1409.1556.pdf) **(VGGNet,神经网络变得十分深!)** :star::star::star:  

**[6]** Szegedy, Christian, et al. "**Going deeper with convolutions**." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) **(GoogLeNet)** :star::star::star:  

**[7]** He, Kaiming, et al. "**Deep residual learning for image recognition**." arXiv preprint arXiv:1512.03385 (2015). [[pdf]](https://arxiv.org/pdf/1512.03385.pdf) **(残差ResNet网络,非常非常深的网络, CVPR最佳论文)** :star::star::star::star::star:  

## 1.4 语音识别的演变  
**[8]** Hinton, Geoffrey, et al. "**Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups**." IEEE Signal Processing Magazine 29.6 (2012): 82-97. [[pdf]](http://cs224d.stanford.edu/papers/maas_paper.pdf) **(语音识别突破)**:star::star::star::star:  

**[9]** Graves, Alex, Abdel-rahman Mohamed, and Geoffrey Hinton. "**Speech recognition with deep recurrent neural networks**." 2013 IEEE international conference on acoustics, speech and signal processing. IEEE, 2013. [[pdf]](http://arxiv.org/pdf/1303.5778.pdf) **(RNN网络)**:star::star::star:  

**[10]** Graves, Alex, and Navdeep Jaitly. "**Towards End-To-End Speech Recognition with Recurrent Neural Networks**." ICML. Vol. 14. 2014. [[pdf]](http://www.jmlr.org/proceedings/papers/v32/graves14.pdf):star::star::star:   

**[11]** Sak, Haşim, et al. "**Fast and accurate recurrent neural network acoustic models for speech recognition**." arXiv preprint arXiv:1507.06947 (2015). [[pdf]](http://arxiv.org/pdf/1507.06947) **(Google语音识别系统)** :star::star::star:  

**[12]** Amodei, Dario, et al. "**Deep speech 2: End-to-end speech recognition in english and mandarin**." arXiv preprint arXiv:1512.02595 (2015). [[pdf]](https://arxiv.org/pdf/1512.02595.pdf) **(百度语音识别系统)** :star::star::star::star:  

**[13]** W. Xiong, J. Droppo, X. Huang, F. Seide, M. Seltzer, A. Stolcke, D. Yu, G. Zweig "**Achieving Human Parity in Conversational Speech Recognition**." arXiv preprint arXiv:1610.05256 (2016). [[pdf]](https://arxiv.org/pdf/1610.05256v1) **(最先进的语音识别, 微软)** :star::star::star::star:  

> 阅读完上述论文后，你将对深度学习历史，深度学习模式（包括CNN，RNN，LSTM）的基本架构以及如何将深度学习应用于图像和语音识别问题有一个基本的了解。 以下文章将深入了解在不同领域的应用和前沿的深度学习方法。我建议你可以根据你的兴趣和研究方向选择以下论文。
# 2 深度学习
## 2.1 模型  
**[14]** Hinton, Geoffrey E., et al. "**Improving neural networks by preventing co-adaptation of feature detectors**." arXiv preprint arXiv:1207.0580 (2012). [[pdf]](https://arxiv.org/pdf/1207.0580.pdf) **(Dropout丢弃法)** :star::star::star:  

**[15]** Srivastava, Nitish, et al. "**Dropout: a simple way to prevent neural networks from overfitting**." Journal of Machine Learning Research 15.1 (2014): 1929-1958. [[pdf]](http://www.jmlr.org/papers/volume15/srivastava14a.old/source/srivastava14a.pdf) :star::star::star:  

**[16]** Ioffe, Sergey, and Christian Szegedy. "**Batch normalization: Accelerating deep network training by reducing internal covariate shift**." arXiv preprint arXiv:1502.03167 (2015). [[pdf]](http://arxiv.org/pdf/1502.03167) **(2015杰出作品)** :star::star::star::star:  

**[17]** Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "**Layer normalization**." arXiv preprint arXiv:1607.06450 (2016). [[pdf]](https://arxiv.org/pdf/1607.06450.pdf?utm_source=sciontist.com&utm_medium=refer&utm_campaign=promote) **(更新批量归一化)** :star::star::star::star:   

**[18]** Courbariaux, Matthieu, et al. "**Binarized Neural Networks: Training Neural Networks with Weights and Activations Constrained to+ 1 or−1**." [[pdf]](https://pdfs.semanticscholar.org/f832/b16cb367802609d91d400085eb87d630212a.pdf) **(新模型，快速)**  :star::star::star:  

**[19]** Jaderberg, Max, et al. "**Decoupled neural interfaces using synthetic gradients**." arXiv preprint arXiv:1608.05343 (2016). [[pdf]](https://arxiv.org/pdf/1608.05343) **(训练方式的创新)** :star::star::star::star::star:  

**[20]** Chen, Tianqi, Ian Goodfellow, and Jonathon Shlens. "Net2net: Accelerating learning via knowledge transfer." arXiv preprint arXiv:1511.05641 (2015). [[pdf]](https://arxiv.org/abs/1511.05641) **(修改前期训练网络以减少训练时期epochs)** :star::star::star:  

**[21]** Wei, Tao, et al. "Network Morphism." arXiv preprint arXiv:1603.01670 (2016). [[pdf]](https://arxiv.org/abs/1603.01670) **(同样，修改以往训练网络以减少训练时期epochs)** :star::star::star:  

## 2.2 优化
**[22]** Sutskever, Ilya, et al. "**On the importance of initialization and momentum in deep learning**." ICML (3) 28 (2013): 1139-1147. [[pdf]](http://www.jmlr.org/proceedings/papers/v28/sutskever13.pdf) **(动量法)** :star::star:  

**[23]** Kingma, Diederik, and Jimmy Ba. "**Adam: A method for stochastic optimization**." arXiv preprint arXiv:1412.6980 (2014). [[pdf]](http://arxiv.org/pdf/1412.6980) **(Adam算法， 也许你经常使用)** :star::star::star:  

**[24]** Andrychowicz, Marcin, et al. "**Learning to learn by gradient descent by gradient descent**." arXiv preprint arXiv:1606.04474 (2016). [[pdf]](https://arxiv.org/pdf/1606.04474) **(神经优化器,非常神奇)** :star::star::star::star::star:  

**[25]** Han, Song, Huizi Mao, and William J. Dally. "**Deep compression: Compressing deep neural network with pruning, trained quantization and huffman coding**." CoRR, abs/1510.00149 2 (2015). [[pdf]](https://pdfs.semanticscholar.org/5b6c/9dda1d88095fa4aac1507348e498a1f2e863.pdf) **(ICLR最佳论文，使NN快速运行的新方向，DeePhi Tech Startup)** :star::star::star::star::star:  

**[26]** Iandola, Forrest N., et al. "**SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 1MB model size**." arXiv preprint arXiv:1602.07360 (2016). [[pdf]](http://arxiv.org/pdf/1602.07360) **(也是优化NN的一个新方向,DeePhi Tech Startup)** :star::star::star::star:  

## 2.3 无监督学习/深度生成模型  
**[27]** Le, Quoc V. "**Building high-level features using large scale unsupervised learning**." 2013 IEEE international conference on acoustics, speech and signal processing. IEEE, 2013. [[pdf]](http://arxiv.org/pdf/1112.6209.pdf&embed) **(里程碑, 吴恩达, 谷歌大脑项目, 识别)** :star::star::star::star:  

**[28]** Kingma, Diederik P., and Max Welling. "**Auto-encoding variational bayes**." arXiv preprint arXiv:1312.6114 (2013). [[pdf]](http://arxiv.org/pdf/1312.6114) **(VAE)** :star::star::star::star:  

**[29]** Goodfellow, Ian, et al. "**Generative adversarial nets**." Advances in Neural Information Processing Systems. 2014. [[pdf]](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) **(GAN生成对抗网络)** :star::star::star::star::star:  

**[30]** Radford, Alec, Luke Metz, and Soumith Chintala. "**Unsupervised representation learning with deep convolutional generative adversarial networks**." arXiv preprint arXiv:1511.06434 (2015). [[pdf]](http://arxiv.org/pdf/1511.06434) **(DCGAN深度卷积对抗神经网络)** :star::star::star::star:  

**[31]** Gregor, Karol, et al. "**DRAW: A recurrent neural network for image generation**." arXiv preprint arXiv:1502.04623 (2015). [[pdf]](http://jmlr.org/proceedings/papers/v37/gregor15.pdf) **(VAE变分自动编码)** :star::star::star::star::star:  

**[32]** Oord, Aaron van den, Nal Kalchbrenner, and Koray Kavukcuoglu. "**Pixel recurrent neural networks**." arXiv preprint arXiv:1601.06759 (2016). [[pdf]](http://arxiv.org/pdf/1601.06759) **(PixelRNN像素递归神经网络)** :star::star::star::star:  

**[33]** Oord, Aaron van den, et al. "Conditional image generation with PixelCNN decoders." arXiv preprint arXiv:1606.05328 (2016). [[pdf]](https://arxiv.org/pdf/1606.05328) **(PixelCNN)** :star::star::star::star:  
## 2.4 循环神经网络 / 序列模型  
**[34]** Graves, Alex. "**Generating sequences with recurrent neural networks**." arXiv preprint arXiv:1308.0850 (2013). [[pdf]](http://arxiv.org/pdf/1308.0850) **(LSTM长短期记忆网络, 非常好的生成效果, RNN的魅力)** :star::star::star::star:  

**[35]** Cho, Kyunghyun, et al. "**Learning phrase representations using RNN encoder-decoder for statistical machine translation**." arXiv preprint arXiv:1406.1078 (2014). [[pdf]](http://arxiv.org/pdf/1406.1078) **(第一篇序列模型论文)** :star::star::star::star:  

**[36]** Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "**Sequence to sequence learning with neural networks**." Advances in neural information processing systems. 2014. [[pdf]](https://arxiv.org/pdf/1409.3215.pdf) **(非常不错的作品)** :star::star::star::star::star:  

**[37]** Bahdanau, Dzmitry, KyungHyun Cho, and Yoshua Bengio. "**Neural Machine Translation by Jointly Learning to Align and Translate**." arXiv preprint arXiv:1409.0473 (2014). [[pdf]](https://arxiv.org/pdf/1409.0473v7.pdf) :star::star::star::star:  

**[38]** Vinyals, Oriol, and Quoc Le. "**A neural conversational model**." arXiv preprint arXiv:1506.05869 (2015). [[pdf]](http://arxiv.org/pdf/1506.05869.pdf%20(http://arxiv.org/pdf/1506.05869.pdf)) **(序列模型在聊天机器人)** :star::star::star:  

## 2.5 神经图灵机  
**[39]** Graves, Alex, Greg Wayne, and Ivo Danihelka. "**Neural turing machines**." arXiv preprint arXiv:1410.5401 (2014). [[pdf]](http://arxiv.org/pdf/1410.5401.pdf) **(未来计算机基本原型)** :star::star::star::star::star:  

**[40]** Zaremba, Wojciech, and Ilya Sutskever. "**Reinforcement learning neural Turing machines**." arXiv preprint arXiv:1505.00521 362 (2015). [[pdf]](https://pdfs.semanticscholar.org/f10e/071292d593fef939e6ef4a59baf0bb3a6c2b.pdf)（强化学习） :star::star::star:  

**[41]** Weston, Jason, Sumit Chopra, and Antoine Bordes. "**Memory networks**." arXiv preprint arXiv:1410.3916 (2014). [[pdf]](http://arxiv.org/pdf/1410.3916) :star::star::star:    

**[42]** Sukhbaatar, Sainbayar, Jason Weston, and Rob Fergus. "**End-to-end memory networks**." Advances in neural information processing systems. 2015. [[pdf]](http://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf) :star::star::star::star:  

**[43]** Vinyals, Oriol, Meire Fortunato, and Navdeep Jaitly. "**Pointer networks**." Advances in Neural Information Processing Systems. 2015. [[pdf]](http://papers.nips.cc/paper/5866-pointer-networks.pdf) :star::star::star::star:  

**[44]** Graves, Alex, et al. "**Hybrid computing using a neural network with dynamic external memory**." Nature (2016). [[pdf]](https://www.dropbox.com/s/0a40xi702grx3dq/2016-graves.pdf) **(里程碑，结合以上论文的想法)** :star::star::star::star::star:  

## 2.6 深度强化学习  
**[45]** Mnih, Volodymyr, et al. "**Playing atari with deep reinforcement learning**." arXiv preprint arXiv:1312.5602 (2013). [[pdf]](http://arxiv.org/pdf/1312.5602.pdf)) **(第一篇命名为深度强化学习文章)** :star::star::star::star:  

**[46]** Mnih, Volodymyr, et al. "**Human-level control through deep reinforcement learning**." Nature 518.7540 (2015): 529-533. [[pdf]](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf) **(里程碑)** :star::star::star::star::star:  

**[47]** Wang, Ziyu, Nando de Freitas, and Marc Lanctot. "**Dueling network architectures for deep reinforcement learning**." arXiv preprint arXiv:1511.06581 (2015). [[pdf]](http://arxiv.org/pdf/1511.06581) **(ICLR 最佳论文)**  :star::star::star::star:   

**[48]** Mnih, Volodymyr, et al. "**Asynchronous methods for deep reinforcement learning**." arXiv preprint arXiv:1602.01783 (2016). [[pdf]](http://arxiv.org/pdf/1602.01783) **(最先进的方法)** :star::star::star::star::star:  

**[49]** Lillicrap, Timothy P., et al. "**Continuous control with deep reinforcement learning**." arXiv preprint arXiv:1509.02971 (2015). [[pdf]](http://arxiv.org/pdf/1509.02971) **(DDPG深度确定策略梯度)** :star::star::star::star:

**[50]** Gu, Shixiang, et al. "**Continuous Deep Q-Learning with Model-based Acceleration**." arXiv preprint arXiv:1603.00748 (2016). [[pdf]](http://arxiv.org/pdf/1603.00748) **(NAF)** :star::star::star::star:

**[51]** Schulman, John, et al. "**Trust region policy optimization**." CoRR, abs/1502.05477 (2015). [[pdf]](http://www.jmlr.org/proceedings/papers/v37/schulman15.pdf) **(TRPO)** :star::star::star::star:

**[52]** Silver, David, et al. "**Mastering the game of Go with deep neural networks and tree search**." Nature 529.7587 (2016): 484-489. [[pdf]](http://willamette.edu/~levenick/cs448/goNature.pdf) **(AlphaGo不用多说！！！)** :star::star::star::star::star:

## 2.7 深度迁移学习 /终身学习 / 

**[53]** Bengio, Yoshua. "**Deep Learning of Representations for Unsupervised and Transfer Learning**." ICML Unsupervised and Transfer Learning 27 (2012): 17-36. [[pdf]](http://www.jmlr.org/proceedings/papers/v27/bengio12a/bengio12a.pdf) **(教程)** :star::star::star:

**[54]** Silver, Daniel L., Qiang Yang, and Lianghao Li. "**Lifelong Machine Learning Systems: Beyond Learning Algorithms**." AAAI Spring Symposium: Lifelong Machine Learning. 2013. [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.696.7800&rep=rep1&type=pdf) **(简短关于终身学习的讨论)**  :star::star::star:

**[55]** Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "**Distilling the knowledge in a neural network**." arXiv preprint arXiv:1503.02531 (2015). [[pdf]](http://arxiv.org/pdf/1503.02531) **(深度学习教父的作品)** :star::star::star::star:

**[56]** Rusu, Andrei A., et al. "**Policy distillation**." arXiv preprint arXiv:1511.06295 (2015). [[pdf]](http://arxiv.org/pdf/1511.06295) **(RL domain)** :star::star::star:

**[57]** Parisotto, Emilio, Jimmy Lei Ba, and Ruslan Salakhutdinov. "**Actor-mimic: Deep multitask and transfer reinforcement learning**." arXiv preprint arXiv:1511.06342 (2015). [[pdf]](http://arxiv.org/pdf/1511.06342) **(RL domain)** :star::star::star:

**[58]** Rusu, Andrei A., et al. "**Progressive neural networks**." arXiv preprint arXiv:1606.04671 (2016). [[pdf]](https://arxiv.org/pdf/1606.04671) **(一个新颖的想法)** :star::star::star::star::star:


## 2.8 一次学习（深度学习）

**[59]** Lake, Brenden M., Ruslan Salakhutdinov, and Joshua B. Tenenbaum. "**Human-level concept learning through probabilistic program induction**." Science 350.6266 (2015): 1332-1338. [[pdf]](http://clm.utexas.edu/compjclub/wp-content/uploads/2016/02/lake2015.pdf) **(不是深度学习，但值得一读)** :star::star::star::star::star:

**[60]** Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov. "**Siamese Neural Networks for One-shot Image Recognition**."(2015) [[pdf]](http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf) :star::star::star:

**[61]** Santoro, Adam, et al. "**One-shot Learning with Memory-Augmented Neural Networks**." arXiv preprint arXiv:1605.06065 (2016). [[pdf]](http://arxiv.org/pdf/1605.06065) **(一次学习（one-shot）基本步骤)** :star::star::star::star:

**[62]** Vinyals, Oriol, et al. "**Matching Networks for One Shot Learning**." arXiv preprint arXiv:1606.04080 (2016). [[pdf]](https://arxiv.org/pdf/1606.04080) :star::star::star:

**[63]** Hariharan, Bharath, and Ross Girshick. "**Low-shot visual object recognition**." arXiv preprint arXiv:1606.02819 (2016). [[pdf]](http://arxiv.org/pdf/1606.02819) **(走向大数据的一步)** :star::star::star::star:


# 3 应用

## 3.1 自然语言处理（NLP）

**[1]** Antoine Bordes, et al. "**Joint Learning of Words and Meaning Representations for Open-Text Semantic Parsing**." AISTATS(2012) [[pdf]](https://www.hds.utc.fr/~bordesan/dokuwiki/lib/exe/fetch.php?id=en%3Apubli&cache=cache&media=en:bordes12aistats.pdf) :star::star::star::star:

**[2]** Mikolov, et al. "**Distributed representations of words and phrases and their compositionality**." ANIPS(2013): 3111-3119 [[pdf]](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) **(word2vec模型)** :star::star::star:

**[3]** Sutskever, et al. "**“Sequence to sequence learning with neural networks**." ANIPS(2014) [[pdf]](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) :star::star::star:

**[4]** Ankit Kumar, et al. "**“Ask Me Anything: Dynamic Memory Networks for Natural Language Processing**." arXiv preprint arXiv:1506.07285(2015) [[pdf]](https://arxiv.org/abs/1506.07285) :star::star::star::star:

**[5]** Yoon Kim, et al. "**Character-Aware Neural Language Models**." NIPS(2015) arXiv preprint arXiv:1508.06615(2015) [[pdf]](https://arxiv.org/abs/1508.06615) :star::star::star::star:

**[6]** Jason Weston, et al. "**Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks**." arXiv preprint arXiv:1502.05698(2015) [[pdf]](https://arxiv.org/abs/1502.05698) **(bAbI)** :star::star::star:

**[7]** Karl Moritz Hermann, et al. "**Teaching Machines to Read and Comprehend**." arXiv preprint arXiv:1506.03340(2015) [[pdf]](https://arxiv.org/abs/1506.03340) **(CNN/DailyMail 填充式问题)** :star::star:

**[8]** Alexis Conneau, et al. "**Very Deep Convolutional Networks for Natural Language Processing**." arXiv preprint arXiv:1606.01781(2016) [[pdf]](https://arxiv.org/abs/1606.01781) **(文本分类最先进的)** :star::star::star:

**[9]** Armand Joulin, et al. "**Bag of Tricks for Efficient Text Classification**." arXiv preprint arXiv:1607.01759(2016) [[pdf]](https://arxiv.org/abs/1607.01759) **(比起最先进稍差点，但速度更快)** :star::star::star:

## 3.2 物体检测

**[1]** Szegedy, Christian, Alexander Toshev, and Dumitru Erhan. "**Deep neural networks for object detection**." Advances in Neural Information Processing Systems. 2013. [[pdf]](http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf) :star::star::star:

**[2]** Girshick, Ross, et al. "**Rich feature hierarchies for accurate object detection and semantic segmentation**." Proceedings of the IEEE conference on computer vision and pattern recognition. 2014. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) **(RCNN模型)** :star::star::star::star::star:

**[3]** He, Kaiming, et al. "**Spatial pyramid pooling in deep convolutional networks for visual recognition**." European Conference on Computer Vision. Springer International Publishing, 2014. [[pdf]](http://arxiv.org/pdf/1406.4729) **(SPPNet基于空间金字塔池化的卷积神经网络物体检测)** :star::star::star::star:

**[4]** Girshick, Ross. "**Fast r-cnn**." Proceedings of the IEEE International Conference on Computer Vision. 2015. [[pdf]](https://pdfs.semanticscholar.org/8f67/64a59f0d17081f2a2a9d06f4ed1cdea1a0ad.pdf) :star::star::star::star:  

**[5]** Ren, Shaoqing, et al. "**Faster R-CNN: Towards real-time object detection with region proposal networks**." Advances in neural information processing systems. 2015. [[pdf]](http://papers.nips.cc/paper/5638-analysis-of-variational-bayesian-latent-dirichlet-allocation-weaker-sparsity-than-map.pdf) :star::star::star::star:

**[6]** Redmon, Joseph, et al. "**You only look once: Unified, real-time object detection**." arXiv preprint arXiv:1506.02640 (2015). [[pdf]](http://homes.cs.washington.edu/~ali/papers/YOLO.pdf) **(YOLO：只需要看一遍,非常实用)** :star::star::star::star::star:

**[7]** Liu, Wei, et al. "**SSD: Single Shot MultiBox Detector**." arXiv preprint arXiv:1512.02325 (2015). [[pdf]](http://arxiv.org/pdf/1512.02325)(SSD模型：单发多框检测器) :star::star::star:

**[8]** Dai, Jifeng, et al. "**R-FCN: Object Detection via
Region-based Fully Convolutional Networks**." arXiv preprint arXiv:1605.06409 (2016). [[pdf]](https://arxiv.org/abs/1605.06409) (R-FCN基于区域的全连接神经网络):star::star::star::star:

**[9]** He, Gkioxari, et al. "**Mask R-CNN**" arXiv preprint arXiv:1703.06870 (2017). [[pdf]](https://arxiv.org/abs/1703.06870) (Mask R-CNN):star::star::star::star:
## 3.3 视觉追踪

**[1]** Wang, Naiyan, and Dit-Yan Yeung. "**Learning a deep compact image representation for visual tracking**." Advances in neural information processing systems. 2013. [[pdf]](http://papers.nips.cc/paper/5192-learning-a-deep-compact-image-representation-for-visual-tracking.pdf) **(第一篇论文使用深度学习进行视觉追踪,DLT Tracker)** :star::star::star:

**[2]** Wang, Naiyan, et al. "**Transferring rich feature hierarchies for robust visual tracking**." arXiv preprint arXiv:1501.04587 (2015). [[pdf]](http://arxiv.org/pdf/1501.04587) **(SO-DLT)** :star::star::star::star:

**[3]** Wang, Lijun, et al. "**Visual tracking with fully convolutional networks**." Proceedings of the IEEE International Conference on Computer Vision. 2015. [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Wang_Visual_Tracking_With_ICCV_2015_paper.pdf) **(FCNT)** :star::star::star::star:

**[4]** Held, David, Sebastian Thrun, and Silvio Savarese. "**Learning to Track at 100 FPS with Deep Regression Networks**." arXiv preprint arXiv:1604.01802 (2016). [[pdf]](http://arxiv.org/pdf/1604.01802) **(GOTURN,Really fast as a deep learning method,but still far behind un-deep-learning methods)** :star::star::star::star:

**[5]** Bertinetto, Luca, et al. "**Fully-Convolutional Siamese Networks for Object Tracking**." arXiv preprint arXiv:1606.09549 (2016). [[pdf]](https://arxiv.org/pdf/1606.09549) **(SiameseFC,New state-of-the-art for real-time object tracking)** :star::star::star::star:

**[6]** Martin Danelljan, Andreas Robinson, Fahad Khan, Michael Felsberg. "**Beyond Correlation Filters: Learning Continuous Convolution Operators for Visual Tracking**." ECCV (2016) [[pdf]](http://www.cvl.isy.liu.se/research/objrec/visualtracking/conttrack/C-COT_ECCV16.pdf) **(C-COT)** :star::star::star::star:

**[7]** Nam, Hyeonseob, Mooyeol Baek, and Bohyung Han. "**Modeling and Propagating CNNs in a Tree Structure for Visual Tracking**." arXiv preprint arXiv:1608.07242 (2016). [[pdf]](https://arxiv.org/pdf/1608.07242) **(VOT2016 Winner,TCNN)** :star::star::star::star:

## 3.4 看图说话
**[1]** Farhadi,Ali,etal. "**Every picture tells a story: Generating sentences from images**". In Computer VisionECCV 2010. Springer Berlin Heidelberg:15-29, 2010. [[pdf]](https://www.cs.cmu.edu/~afarhadi/papers/sentence.pdf) :star::star::star:

**[2]** Kulkarni, Girish, et al. "**Baby talk: Understanding and generating image descriptions**". In Proceedings of the 24th CVPR, 2011. [[pdf]](http://tamaraberg.com/papers/generation_cvpr11.pdf):star::star::star::star:

**[3]** Vinyals, Oriol, et al. "**Show and tell: A neural image caption generator**". In arXiv preprint arXiv:1411.4555, 2014. [[pdf]](https://arxiv.org/pdf/1411.4555.pdf):star::star::star:

**[4]** Donahue, Jeff, et al. "**Long-term recurrent convolutional networks for visual recognition and description**". In arXiv preprint arXiv:1411.4389 ,2014. [[pdf]](https://arxiv.org/pdf/1411.4389.pdf)

**[5]** Karpathy, Andrej, and Li Fei-Fei. "**Deep visual-semantic alignments for generating image descriptions**". In arXiv preprint arXiv:1412.2306, 2014. [[pdf]](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf):star::star::star::star::star:

**[6]** Karpathy, Andrej, Armand Joulin, and Fei Fei F. Li. "**Deep fragment embeddings for bidirectional image sentence mapping**". In Advances in neural information processing systems, 2014. [[pdf]](https://arxiv.org/pdf/1406.5679v1.pdf):star::star::star::star:

**[7]** Fang, Hao, et al. "**From captions to visual concepts and back**". In arXiv preprint arXiv:1411.4952, 2014. [[pdf]](https://arxiv.org/pdf/1411.4952v3.pdf):star::star::star::star::star:

**[8]** Chen, Xinlei, and C. Lawrence Zitnick. "**Learning a recurrent visual representation for image caption generation**". In arXiv preprint arXiv:1411.5654, 2014. [[pdf]](https://arxiv.org/pdf/1411.5654v1.pdf):star::star::star::star:

**[9]** Mao, Junhua, et al. "**Deep captioning with multimodal recurrent neural networks (m-rnn)**". In arXiv preprint arXiv:1412.6632, 2014. [[pdf]](https://arxiv.org/pdf/1412.6632v5.pdf):star::star::star:

**[10]** Xu, Kelvin, et al. "**Show, attend and tell: Neural image caption generation with visual attention**". In arXiv preprint arXiv:1502.03044, 2015. [[pdf]](https://arxiv.org/pdf/1502.03044v3.pdf):star::star::star::star::star:

## 3.5 机器翻译

> 以下列的是一些RNN相关的 / 序列模型.

**[1]** Luong, Minh-Thang, et al. "**Addressing the rare word problem in neural machine translation**." arXiv preprint arXiv:1410.8206 (2014). [[pdf]](http://arxiv.org/pdf/1410.8206) :star::star::star::star:


**[2]** Sennrich, et al. "**Neural Machine Translation of Rare Words with Subword Units**". In arXiv preprint arXiv:1508.07909, 2015. [[pdf]](https://arxiv.org/pdf/1508.07909.pdf):star::star::star:

**[3]** Luong, Minh-Thang, Hieu Pham, and Christopher D. Manning. "**Effective approaches to attention-based neural machine translation**." arXiv preprint arXiv:1508.04025 (2015). [[pdf]](http://arxiv.org/pdf/1508.04025) :star::star::star::star:

**[4]** Chung, et al. "**A Character-Level Decoder without Explicit Segmentation for Neural Machine Translation**". In arXiv preprint arXiv:1603.06147, 2016. [[pdf]](https://arxiv.org/pdf/1603.06147.pdf):star::star:

**[5]** Lee, et al. "**Fully Character-Level Neural Machine Translation without Explicit Segmentation**". In arXiv preprint arXiv:1610.03017, 2016. [[pdf]](https://arxiv.org/pdf/1610.03017.pdf):star::star::star::star::star:

**[6]** Wu, Schuster, Chen, Le, et al. "**Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation**". In arXiv preprint arXiv:1609.08144v2, 2016. [[pdf]](https://arxiv.org/pdf/1609.08144v2.pdf) **(Milestone)** :star::star::star::star:

## 3.6 机器人技术

**[1]** Koutník, Jan, et al. "**Evolving large-scale neural networks for vision-based reinforcement learning**." Proceedings of the 15th annual conference on Genetic and evolutionary computation. ACM, 2013. [[pdf]](http://repository.supsi.ch/4550/1/koutnik2013gecco.pdf) :star::star::star:

**[2]** Levine, Sergey, et al. "**End-to-end training of deep visuomotor policies**." Journal of Machine Learning Research 17.39 (2016): 1-40. [[pdf]](http://www.jmlr.org/papers/volume17/15-522/15-522.pdf) :star::star::star::star::star:

**[3]** Pinto, Lerrel, and Abhinav Gupta. "**Supersizing self-supervision: Learning to grasp from 50k tries and 700 robot hours**." arXiv preprint arXiv:1509.06825 (2015). [[pdf]](http://arxiv.org/pdf/1509.06825) :star::star::star:

**[4]** Levine, Sergey, et al. "**Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection**." arXiv preprint arXiv:1603.02199 (2016). [[pdf]](http://arxiv.org/pdf/1603.02199) :star::star::star::star:

**[5]** Zhu, Yuke, et al. "**Target-driven Visual Navigation in Indoor Scenes using Deep Reinforcement Learning**." arXiv preprint arXiv:1609.05143 (2016). [[pdf]](https://arxiv.org/pdf/1609.05143) :star::star::star::star:

**[6]** Yahya, Ali, et al. "**Collective Robot Reinforcement Learning with Distributed Asynchronous Guided Policy Search**." arXiv preprint arXiv:1610.00673 (2016). [[pdf]](https://arxiv.org/pdf/1610.00673) :star::star::star::star:

**[7]** Gu, Shixiang, et al. "**Deep Reinforcement Learning for Robotic Manipulation**." arXiv preprint arXiv:1610.00633 (2016). [[pdf]](https://arxiv.org/pdf/1610.00633) :star::star::star::star:

**[8]** A Rusu, M Vecerik, Thomas Rothörl, N Heess, R Pascanu, R Hadsell."**Sim-to-Real Robot Learning from Pixels with Progressive Nets**." arXiv preprint arXiv:1610.04286 (2016). [[pdf]](https://arxiv.org/pdf/1610.04286.pdf) :star::star::star::star:

**[9]** Mirowski, Piotr, et al. "**Learning to navigate in complex environments**." arXiv preprint arXiv:1611.03673 (2016). [[pdf]](https://arxiv.org/pdf/1611.03673) :star::star::star::star:

## 3.7 艺术方面

**[1]** Mordvintsev, Alexander; Olah, Christopher; Tyka, Mike (2015). "**Inceptionism: Going Deeper into Neural Networks**". Google Research. [[html]](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html) **(Deep Dream)**
:star::star::star::star:

**[2]** Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "**A neural algorithm of artistic style**." arXiv preprint arXiv:1508.06576 (2015). [[pdf]](http://arxiv.org/pdf/1508.06576) **(Outstanding Work, 目前最成功的方法)** :star::star::star::star::star:

**[3]** Zhu, Jun-Yan, et al. "**Generative Visual Manipulation on the Natural Image Manifold**." European Conference on Computer Vision. Springer International Publishing, 2016. [[pdf]](https://arxiv.org/pdf/1609.03552) **(iGAN)** :star::star::star::star:

**[4]** Champandard, Alex J. "**Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artworks**." arXiv preprint arXiv:1603.01768 (2016). [[pdf]](http://arxiv.org/pdf/1603.01768) **(神经涂鸦)** :star::star::star::star:

**[5]** Zhang, Richard, Phillip Isola, and Alexei A. Efros. "**Colorful Image Colorization**." arXiv preprint arXiv:1603.08511 (2016). [[pdf]](http://arxiv.org/pdf/1603.08511) :star::star::star::star:

**[6]** Johnson, Justin, Alexandre Alahi, and Li Fei-Fei. "**Perceptual losses for real-time style transfer and super-resolution**." arXiv preprint arXiv:1603.08155 (2016). [[pdf]](https://arxiv.org/pdf/1603.08155.pdf) :star::star::star::star:

**[7]** Vincent Dumoulin, Jonathon Shlens and Manjunath Kudlur. "**A learned representation for artistic style**." arXiv preprint arXiv:1610.07629 (2016). [[pdf]](https://arxiv.org/pdf/1610.07629v1.pdf) :star::star::star::star:

**[8]** Gatys, Leon and Ecker, et al."**Controlling Perceptual Factors in Neural Style Transfer**." arXiv preprint arXiv:1611.07865 (2016). [[pdf]](https://arxiv.org/pdf/1611.07865.pdf) **(控制风格转移的空间位置，颜色信息和跨空间尺度)**:star::star::star::star:

**[9]** Ulyanov, Dmitry and Lebedev, Vadim, et al. "**Texture Networks: Feed-forward Synthesis of Textures and Stylized Images**." arXiv preprint arXiv:1603.03417(2016). [[pdf]](http://arxiv.org/abs/1603.03417) **(纹理生成和风格转移)** :star::star::star::star:


## 3.8 对象分割

**[1]** J. Long, E. Shelhamer, and T. Darrell, “**Fully convolutional networks for semantic segmentation**.” in CVPR, 2015. [[pdf]](https://arxiv.org/pdf/1411.4038v2.pdf) :star::star::star::star::star:

**[2]** L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille. "**Semantic image segmentation with deep convolutional nets and fully connected crfs**." In ICLR, 2015. [[pdf]](https://arxiv.org/pdf/1606.00915v1.pdf) :star::star::star::star::star:

**[3]** Pinheiro, P.O., Collobert, R., Dollar, P. "**Learning to segment object candidates.**" In: NIPS. 2015. [[pdf]](https://arxiv.org/pdf/1506.06204v2.pdf) :star::star::star::star:

**[4]** Dai, J., He, K., Sun, J. "**Instance-aware semantic segmentation via multi-task network cascades**." in CVPR. 2016 [[pdf]](https://arxiv.org/pdf/1512.04412v1.pdf) :star::star::star:

**[5]** Dai, J., He, K., Sun, J. "**Instance-sensitive Fully Convolutional Networks**." arXiv preprint arXiv:1603.08678 (2016). [[pdf]](https://arxiv.org/pdf/1603.08678v1.pdf) :star::star::star:
