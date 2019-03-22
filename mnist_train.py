# coding=utf-8
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.9
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

#模型保存路径
MODEL_SAVE_PATH = "/Users/qinchao/Desktop/MNIST/model/"
MODEL_NAME = 'model.ckpt'

def train(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False) #统计次数

    #滑动平均
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())#对所有可训练的变量执行滑动平均

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1)) #对每个样本计算损失

    cross_entropy_mean = tf.reduce_mean(cross_entropy) #求batch的平均损失
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses')) #总的损失

    #训练
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                        mnist.train.num_examples/BATCH_SIZE,
                        LEARNING_RATE_DECAY) #每整体遍历一遍样本学习率就乘以衰减因子
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    #将loss训练和滑动平均同时优化
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')

    #tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run() #所有变量初始化

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value,step = sess.run([train_op, loss, global_step], feed_dict=
                                            {x:xs,y_:ys }) #每执行一次，global_step +=1

            if i % 1000 == 0:
                print('After %d trainning step(s), loss on trainning batch is %g' % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv = None):
    mnist = input_data.read_data_sets("/Users/qinchao/Desktop/MNIST/mnist_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()

