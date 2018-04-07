#-*-coding:utf-8-*- 
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
print("MNIST ready")

n_input  = 784 # 28*28的灰度图，像素个数784
n_output = 10  # 是10分类问题

# 权重项
weights = {
    # conv1,参数[3, 3, 1, 32]分别指定了filter的h、w、所连接输入的维度、filter的个数即产生特征图个数
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.1)),   
    # conv2，这里参数3，3同上，32是当前连接的深度是32，即前面特征图的个数，64为输出的特征图的个数
    'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.1)), 
    # fc1，将特征图转换为向量，1024由自己定义
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024], stddev=0.1)), 
    # fc2，做10分类任务，前面连1024，输出10分类
    'wd2': tf.Variable(tf.random_normal([1024, n_output], stddev=0.1)) 
}
"""
特征图大小计算：
f_w = (w-f+2*pad)/s + 1 = (28-3+2*1)/1 + 1 = 28 # 说明经过卷积层并没有改变图片的大小
f_h = (h-f+2*pad)/s + 1 = (28-3+2*1)/1 + 1 = 28
# 特征图的大小是经过池化层后改变的
第一次pooling后28*28变为14*14
第二次pooling后14*14变为7*7，即最终是一个7*7*64的特征图

"""
# 偏置项
biases = {
    'bc1': tf.Variable(tf.random_normal([32], stddev=0.1)),      # conv1，对应32个特征图
    'bc2': tf.Variable(tf.random_normal([64], stddev=0.1)),      # conv2，对应64个特征图
    'bd1': tf.Variable(tf.random_normal([1024], stddev=0.1)),    # fc1，对应1024个向量
    'bd2': tf.Variable(tf.random_normal([n_output], stddev=0.1)) # fc2，对应10个输出
}

def conv_basic(_input, _w, _b, _keep_prob):
    # INPUT
    # 对图像做预处理，转换为tf支持的格式，即[n, h, w, c],-1是确定好其它3维后，让tf去推断剩下的1维
    _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1]) 

    # CONV LAYER 1
    _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME') 
    # [1, 1, 1, 1]分别代表batch_size、h、w、c的stride
    # padding有两种选择：'SAME'（窗口滑动时，像素不够会自动补0）或'VALID'（不够就跳过）两种选择
    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1'])) # 卷积层后连激活函数
    # 最大值池化，[1, 2, 2, 1]其中1,1对应batch_size和channel，2,2对应2*2的池化
    _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 随机杀死一些神经元,_keepratio为保留神经元比例，如0.6 
    _pool_dr1 = tf.nn.dropout(_pool1, _keep_prob) 

    # CONV LAYER 2
    _conv2 = tf.nn.conv2d(_pool_dr1, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
    _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr2 = tf.nn.dropout(_pool2, _keep_prob) # dropout

    # VECTORIZE向量化
    # 定义全连接层的输入，把pool2的输出做一个reshape，变为向量的形式
    _densel = tf.reshape(_pool_dr2, [-1, _w['wd1'].get_shape().as_list()[0]]) 

    # FULLY CONNECTED LAYER 1
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_densel, _w['wd1']), _b['bd1'])) # w*x+b，再通过relu
    _fc_dr1 = tf.nn.dropout(_fc1, _keep_prob) # dropout

    # FULLY CONNECTED LAYER 2
    _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2']) # w*x+b，得到结果

    # RETURN
    out = {'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1, 'pool_dr1': _pool_dr1,
           'conv2': _conv2, 'pool2': _pool2, 'pool_dr2': _pool_dr2, 'densel': _densel,
           'fc1': _fc1, 'fc_dr1': _fc_dr1, 'out': _out
           }
    return out
print("CNN READY")


x = tf.placeholder(tf.float32, [None, n_input]) # 用placeholder先占地方，样本个数不确定为None
y = tf.placeholder(tf.float32, [None, n_output]) # 用placeholder先占地方，样本个数不确定为None
keep_prob = tf.placeholder(tf.float32)

_pred = conv_basic(x, weights, biases, keep_prob)['out'] # 前向传播的预测值
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=_pred, labels=y)) # 交叉熵损失函数
optm = tf.train.AdamOptimizer(0.001).minimize(cost) # 梯度下降优化器
_corr = tf.equal(tf.argmax(_pred, 1), tf.argmax(y, 1)) # 对比预测值索引和实际label索引，相同返回True，不同返回False
accr = tf.reduce_mean(tf.cast(_corr, tf.float32)) # 将True或False转换为1或0,并对所有的判断结果求均值

init = tf.global_variables_initializer()
print("FUNCTIONS READY")

# 上面神经网络结构定义好之后，下面定义一些超参数
training_epochs = 1000 # 所有样本迭代1000次
batch_size = 100 # 每进行一次迭代选择100个样本
display_step = 1
# LAUNCH THE GRAPH
sess = tf.Session() # 定义一个Session
sess.run(init) # 在sess里run一下初始化操作
# OPTIMIZE
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size) # 逐个batch的去取数据
        sess.run(optm, feed_dict={x: batch_xs, y: batch_ys, keep_prob:0.5})
        avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob:1.0})/total_batch
    if epoch % display_step == 0:
        train_accuracy = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        test_accuracy = sess.run(accr, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob:1.0})
        print("Epoch: %03d/%03d cost: %.9f TRAIN ACCURACY: %.3f TEST ACCURACY: %.3f"
              % (epoch, training_epochs, avg_cost, train_accuracy, test_accuracy))
print("DONE")