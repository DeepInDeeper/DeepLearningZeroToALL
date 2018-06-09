import numpy as np 
import matplotlib as mp

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import math
import os
import datetime

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tf.reset_default_graph()
batchSize = 128
learning_rate = 0.001
OUTPUT_DIR = "tmp"
logs_path = os.path.join(
            OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
if not os.path.exists(logs_path):
    os.makedirs(logs_path)


x = tf.placeholder(tf.float32, [None, 784],name="x-in")
true_y = tf.placeholder(tf.float32, [None, 10],name="y-in")
keep_prob = tf.placeholder("float")

x_image = tf.reshape(x,[-1,28,28,1])
tf.summary.image('input', x_image, 3)



hidden_1 = slim.conv2d(x_image,32,[3,3],scope="hidden_1")
pool_1 = slim.max_pool2d(hidden_1,[2,2],scope="pool_1")

hidden_2 = slim.conv2d(pool_1,64,[3,3],scope="hidden_2")
pool_2 = slim.max_pool2d(hidden_2,[2,2],scope="pool_2")


out_y = slim.fully_connected(slim.flatten(pool_2),10,activation_fn=tf.nn.softmax,scope="out_y")

with tf.name_scope('Accuracy'):
	correct_prediction = tf.equal(tf.argmax(out_y,1), tf.argmax(true_y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# specify cost function
with tf.name_scope('cross_entropy'):
	cross_entropy = -tf.reduce_sum(true_y*tf.log(out_y))

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,name="Adam")


# create a summary for our cost and accuracy
tf.summary.scalar("cost", cross_entropy)
tf.summary.scalar("accuracy", accuracy)

# merge all summaries into a single "operation" which we can execute in a session 
summary_op = tf.summary.merge_all()
saver = tf.train.Saver()

with tf.Session() as sess:
	# variables need to be initialized before we can use them
	sess.run(tf.global_variables_initializer())
	# create log writer object
	writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

	for i in range(1001):
	    batch = mnist.train.next_batch(batchSize)
	    _,summary = sess.run([train_step,summary_op], feed_dict={x:batch[0],true_y:batch[1], keep_prob:0.5})
	    writer.add_summary(summary,i)
	    if i % 100 == 0 and i != 0:
	        trainAccuracy = sess.run(accuracy, feed_dict={x:batch[0],true_y:batch[1], keep_prob:1.0})
	        saver.save(sess,"tmp/my-model",i)
	        print("step %d, training accuracy %g"%(i, trainAccuracy))

	testAccuracy = sess.run(accuracy, feed_dict={x:mnist.test.images[:256],true_y:mnist.test.labels[:256], keep_prob:1.0})
	print("test accuracy %g"%(testAccuracy))


	Use = mnist.test.images[10]
	plt.imshow(np.reshape(Use,[28,28]), interpolation="nearest", cmap="gray")
	r = 10
	print("Prediction: ", sess.run(
	    tf.argmax(out_y, 1), feed_dict={x: mnist.test.images[r:r + 1]}))

	print("Probs: ", sess.run(
	    out_y, feed_dict={x: mnist.test.images[r:r + 1]}))
