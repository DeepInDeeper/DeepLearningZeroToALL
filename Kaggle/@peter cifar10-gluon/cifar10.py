# -*- coding: utf-8 -*- 

import pre_deal,model  
from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon.data import vision
import numpy as np
from mxnet.gluon import nn
from matplotlib import pyplot as plt
from utils import Visualizer

train_dir = 'train'
test_dir = 'test'
batch_size = 128
data_dir = '/media/yijie/娱乐/tmp/kaggle_cifar10'
label_file = 'D:/dataset/gluon/train_valid_test/trainLabels.csv'
input_dir = 'D:/dataset/gluon/train_valid_test'
valid_ratio = 0.1
pre_deal_flag = True

vis = Visualizer(env='CIFAR10')

# sorting the dataset and transform
if not pre_deal_flag:
    pre_deal.reorg_cifar10_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio)
    
input_str = input_dir + '/'


# 读取原始图像文件。flag=1说明输入图像有三个通道（彩色）。
train_ds = vision.ImageFolderDataset(input_str + 'train', flag=1, 
                                 transform=pre_deal.transform_train)
valid_ds = vision.ImageFolderDataset(input_str + 'valid', flag=1, 
                                 transform=pre_deal.transform_test)
train_valid_ds = vision.ImageFolderDataset(input_str + 'train_valid', 
                                       flag=1, transform=pre_deal.transform_train)
test_ds = vision.ImageFolderDataset(input_str + 'test', flag=1, 
                                 transform=pre_deal.transform_test)

loader = gluon.data.DataLoader
train_data = loader(train_ds, batch_size, shuffle=True, last_batch='keep')
valid_data = loader(valid_ds, batch_size, shuffle=True, last_batch='keep')
train_valid_data = loader(train_valid_ds, batch_size, shuffle=True, last_batch='keep')
test_data = loader(test_ds, batch_size, shuffle=False, last_batch='keep')


# 交叉熵损失函数。
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()




import datetime
import sys
sys.path.append('..')
import utils


def train(net, train_data, valid_data, num_epochs, lr, wd, ctx, lr_period, lr_decay):
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})

    prev_time = datetime.datetime.now()
    plt_train_acc = []
    plt_valid_acc = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        if epoch > 161 and epoch % 10 == 0:
            trainer.set_learning_rate(trainer.learning_rate * 0.4)
        for data, label in train_data:
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_acc = utils.evaluate_accuracy(valid_data, net, ctx)
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, Valid acc %f, "
                         % (epoch, train_loss / len(train_data),
                            train_acc / len(train_data), valid_acc))
            plt_train_acc.append(train_acc / len(train_data))
            plt_valid_acc.append(valid_acc)
        else:
            epoch_str = ("Epoch %d. Loss: %f, Train acc %f, "
                         % (epoch, train_loss / len(train_data),
                            train_acc / len(train_data)))

        prev_time = cur_time

        print(epoch_str + time_str + ', lr ' + str(trainer.learning_rate))
    # plot 
    if valid_data is not None:
        plt.plot(plt_train_acc)
        plt.plot(plt_valid_acc)
        plt.legend(['train_acc','test_acc'])
        plt.savefig("Loss.png")




ctx = utils.try_gpu()
num_epochs = 200
learning_rate = 0.1
weight_decay = 5e-4
lr_period = 80
lr_decay = 0.1

net = model.get_net(ctx)
net.hybridize()
train(net, train_data, valid_data, num_epochs, learning_rate, 
      weight_decay, ctx, lr_period, lr_decay)

import numpy as np
import pandas as pd

net = model.get_net(ctx)
net.hybridize()
train(net, train_valid_data, None, num_epochs, learning_rate, 
      weight_decay, ctx, lr_period, lr_decay)

preds = []
for data, label in test_data:
    output = net(data.as_in_context(ctx))
    preds.extend(output.argmax(axis=1).astype(int).asnumpy())

sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key = lambda x:str(x))

df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
df.to_csv('submission.csv', index=False)
