 	# 加载数据分析常用库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gzip
import paddle.v2 as paddle
import os
% matplotlib inline

TRAINING_METAPATH = "/home/kesci/work/Broad/INFO/training.json"
VALIDATION_METAPATH = "/home/kesci/work/Broad/INFO/validation.json"
TESTING_METAPATH = "/home/kesci/work/Broad/INFO/testing.json"

train_info = pd.read_json(TRAINING_METAPATH)
valid_info = pd.read_json(VALIDATION_METAPATH)
test_info = pd.read_json(TESTING_METAPATH)
print (train_info.shape,valid_info.shape,test_info.shape)

train_database = train_info["database"]
valid_database = valid_info["database"]
test_database = test_info["database"]

def generater_dataset(file_name,dataset_folder):
    if dataset_folder == "train": 
        DATA_DIR = "/mnt/BROAD-datasets/video/training/"
        database = train_database
    elif dataset_folder == "valid":
        DATA_DIR = "/mnt/BROAD-datasets/video/validation/"
        database = valid_database
    elif dataset_folder == "test":
        DATA_DIR = "/mnt/BROAD-datasets/video/testing/"
    file_name_path = os.path.join(DATA_DIR,str(file_name)+".pkl")
    X = np.load(file_name_path)
    X = np.array(X)
   
    
    
    ## get Y label
    valid_new_list = []
    for valid_list in database[file_name]["annotations"]:
        valid_new_list = valid_new_list + valid_list['segment']
    # 转换为 整数
    valid_new_list = np.array(valid_new_list,dtype="int")
    
    label = np.zeros((X.shape[0],1),dtype=np.int8)
    for i in range(len(valid_new_list)/2):
        label[valid_new_list[2*i]:valid_new_list[2*i+1]] =1
    # 采样10
    X = X[::10,:]
    label = label[::10]
    return X,label

x_train,y_train = generater_dataset("1675d2b39251576a5224f852ab033737","train")
print x_train.shape
plt.plot(y_train)

def concat_data (concat_data):
    if concat_data == "train":
        database = train_database
    elif concat_data == "valid":
        database = valid_database
    elif concat_data == "test":
        database = test_database
        
    a = database.keys()
    x_train,y_train = generater_dataset(a[0],concat_data)
    for i in range(1,len(a)):
        x_train_tmp,y_train_tmp=generater_dataset(a[i],concat_data)
        x_train = np.concatenate((x_train,x_train_tmp),axis=0)
        y_train = np.concatenate((y_train,y_train_tmp),axis=0)
    
    return x_train,y_train


# 将所有的数据可以加载进去，但是按照采样的10 进行的
x_train,y_train = concat_data("train")
x_valid,y_valid = concat_data("valid")
print x_train.shape,y_train.shape


def multilayer_perceptron(img):
    # 第一个全连接层，激活函数为ReLU
    hidden0 = paddle.layer.fc(input=img, size=256, act=paddle.activation.Relu())
    drop0 = paddle.layer.dropout(input=hidden0, dropout_rate=0.5)
    hidden1 = paddle.layer.fc(input=drop0, size=128, act=paddle.activation.Relu())
    # 第二个全连接层，激活函数为ReLU
    hidden2 = paddle.layer.fc(input=hidden1,size=64,act=paddle.activation.Relu())
    drop1 = paddle.layer.dropout(input=hidden2, dropout_rate=0.8)
    # 以softmax为激活函数的全连接输出层，输出层的大小必须为数字的个数10
    predict = paddle.layer.fc(input=hidden2,size=2,act=paddle.activation.Softmax())
    return predict


# 该模型运行在单个CPU上
paddle.init(use_gpu=False, trainer_count=1)

images = paddle.layer.data(
    name='images', type=paddle.data_type.dense_vector(2048))
label = paddle.layer.data(
    name='y', type=paddle.data_type.integer_value(2))

#predict = softmax_regression(images) # Softmax回归
predict = multilayer_perceptron(images) #多层感知器
#predict = convolutional_neural_network(images) #LeNet5卷积神经网络

cost = paddle.layer.classification_cost(input=predict, label=label)

parameters = paddle.parameters.create(cost)

optimizer = paddle.optimizer.Momentum(learning_rate=0.1 / 128.0, momentum=0.9, 
                                      regularization=paddle.optimizer.L2Regularization(rate=0.0005 * 128))

trainer = paddle.trainer.SGD(cost=cost, parameters=parameters, update_equation=optimizer)

from paddle.v2.plot import Ploter

train_title = "Train cost"
test_title = "Valid cost"
cost_ploter = Ploter(train_title, test_title)

step = 0

# event_handler to plot a figure
def event_handler_plot(event):
    global step
    if isinstance(event, paddle.event.EndIteration):
        if step % 100 == 0:
            cost_ploter.append(train_title, step, event.cost)
            cost_ploter.plot()
        step += 1
    if isinstance(event, paddle.event.EndPass):
        # save parameters
        with gzip.open('params_pass_%d.tar.gz' % event.pass_id, 'w') as f:
            parameters.to_tar(f)

        result = trainer.test(reader=paddle.batch(
            reader_creator(x_valid,y_valid), batch_size=256))
        cost_ploter.append(test_title, step, result.cost)



lists = []

def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 100 == 0:
            print "Pass %d, Batch %d, Cost %f, %s" % (
                event.pass_id, event.batch_id, event.cost, event.metrics)
    if isinstance(event, paddle.event.EndPass):
        # save parameters
        with gzip.open('params_pass_%d.tar.gz' % event.pass_id, 'w') as f:
            parameters.to_tar(f)

        result = trainer.test(reader=paddle.batch(
            reader_creator(x_valid,y_valid), batch_size=16))
        print "Test with Pass %d, Cost %f, %s\n" % (
            event.pass_id, result.cost, result.metrics)
        lists.append((event.pass_id, result.cost,
                      result.metrics['classification_error_evaluator']))



def reader_creator(data,label):
    def reader():
        for i in xrange(len(data)):
            yield data[i,:],int(label[i])
#             x = x_train[i].reshape((1,2048))
#             y = y_train[i].reshape((1,1))
#             yield x,y
    return reader


train_reader = reader=paddle.batch(paddle.reader.shuffle(
            reader_creator(x_train,y_train), buf_size=5000),
                                   batch_size=512)
trainer.train(reader=train_reader,num_passes=10,event_handler=event_handler_plot)

# 预测
probs = paddle.infer(output_layer=predict,parameters=parameters,input=x_valid)

## 取出部分用来查看
from matplotlib.lines import Line2D

figure, ax = plt.subplots()
ax.add_line(Line2D((0,200), (0.5,0.5), linewidth=1, color='red'))
#plt.plot(1-probs[:,0].T+probs[:,1].T)
ax.plot(probs[200:400,1])
#ax.plot(probs[200:400,0])
ax.plot(y_valid[200:400])