#-*-coding:utf-8-*- 
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

print('Download and Extract MNIST dataset')
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True) # one_hot=True意思是编码格式为01编码
print("tpye of 'mnist' is %s" % (type(mnist)))
print("number of train data is %d" % (mnist.train.num_examples))
print("number of test data is %d" % (mnist.test.num_examples))
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels
print("MNIST loaded")

"""
print("type of 'trainimg' is %s"    % (type(trainimg)))
print("type of 'trainlabel' is %s"  % (type(trainlabel)))
print("type of 'testimg' is %s"     % (type(testimg)))
print("type of 'testlabel' is %s"   % (type(testlabel)))
print("------------------------------------------------")
print("shape of 'trainimg' is %s"   % (trainimg.shape,))
print("shape of 'trainlabel' is %s" % (trainlabel.shape,))
print("shape of 'testimg' is %s"    % (testimg.shape,))
print("shape of 'testlabel' is %s"  % (testlabel.shape,))

"""

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10]) # None is for infinite
w = tf.Variable(tf.zeros([784, 10])) # 为了方便直接用0初始化，可以高斯初始化
b = tf.Variable(tf.zeros([10])) # 10分类的任务，10种label，所以只需要初始化10个b

pred = tf.nn.softmax(tf.matmul(x, w) + b) # 前向传播的预测值
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=[1])) # 交叉熵损失函数
optm = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)) # tf.equal()对比预测值的索引和真实label的索引是否一样，一样返回True，不一样返回False
accr = tf.reduce_mean(tf.cast(corr, tf.float32))

init = tf.global_variables_initializer() # 全局参数初始化器

training_epochs = 100 # 所有样本迭代100次
batch_size = 100 # 每进行一次迭代选择100个样本
display_step = 5
# SESSION
sess = tf.Session() # 定义一个Session
sess.run(init) # 在sess里run一下初始化操作
# MINI-BATCH LEARNING
for epoch in range(training_epochs): # 每一个epoch进行循环
    avg_cost = 0. # 刚开始损失值定义为0
    num_batch = int(mnist.train.num_examples/batch_size)
    for i in range(num_batch): # 每一个batch进行选择
        batch_xs, batch_ys = mnist.train.next_batch(batch_size) # 通过next_batch()就可以一个一个batch的拿数据，
        sess.run(optm, feed_dict={x: batch_xs, y: batch_ys}) # run一下用梯度下降进行求解，通过placeholder把x，y传进来
        avg_cost += sess.run(cost, feed_dict={x: batch_xs, y:batch_ys})/num_batch
    # DISPLAY
    if epoch % display_step == 0: # display_step之前定义为5，这里每5个epoch打印一下
        train_acc = sess.run(accr, feed_dict={x: batch_xs, y:batch_ys})
        test_acc = sess.run(accr, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Epoch: %03d/%03d cost: %.9f TRAIN ACCURACY: %.3f TEST ACCURACY: %.3f"
              % (epoch, training_epochs, avg_cost, train_acc, test_acc))
print("DONE")