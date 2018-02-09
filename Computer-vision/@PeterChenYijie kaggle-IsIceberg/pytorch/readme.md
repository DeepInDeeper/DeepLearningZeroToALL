update.....
# Pytorch Baseline Code（GPU）
kaggle比赛[Iceberg-classifier-challenge](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge)代码



## 使用
### 1.环境配置
安装：
- [PyTorch](http://pytorch.org/),根据官网说明下载指定的版本即可。
- 第三方依赖： |fire|visdom|sklearn|pandas|numpy|


### 2.启动visdom
可视化工具[visdom]用于显示(https://github.com/facebookresearch/visdom)
    > 国内网站可能在pip的时候有一些会下载不下来，需要自行下载,参考链接[issues185](https://github.com/facebookresearch/visdom/issues/185)下载文件到static下面

```bash
nohup python -m visdom.server&
```

### 3.训练样本并生成提交文件

    python -B main.py train (--model= model = "SimpleNet" --batch_size=256 -epoch=10) 
在kaggle提交能够到0.2048


## jupyter notebook
运行   
       pytorch_Iceberg_classifier.ipynb



主要内容：   
    -- model diff_net.py 存放模型;        
    -- Iceberg_dataset.py 数据前期处理 readSuffleData、getTrainValLoaders   
    -- iceberg_classfier.py generateSingleModel模型训练、testModel给出预测   

### 其他
kaggle的discussion有提到通过对获得的多个csv文件进行不同ensemble，从而降低loss，能够完成0.14几，见essemble.ipynb（省略数据集）
