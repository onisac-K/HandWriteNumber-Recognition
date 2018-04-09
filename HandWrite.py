#encoding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

max_steps = 20000
learning_rate = 1e-4
REGULARIZATION_RATE = 0.0001
dropout = 0.9
data_dir = './MNIST_data'
log_dir = './MNIST_logs'
Save_dir = './MNIST_Model'

#default regularizer
#regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

# get the data
mnist = input_data.read_data_sets(data_dir,one_hot = True)

# create the session
sess = tf.InteractiveSession()

# input layer
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

# reshape
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# initialize the weight
def weight_variable(shape, regularizer):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    weights = tf.Variable(initial)
    if regularizer != None:
       tf.add_to_collection('losses', regularizer(weights))
    return weights

# initialize the bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# summaries
'''
def variable_summaries(var):
    with tf.name_scope('summaries'):
        # 计算参数的均值，并使用tf.summary.scaler记录
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        # 计算参数的标准差
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # 用直方图记录参数的分布
        tf.summary.histogram('histogram', var)
'''

# neural network
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    # 设置命名空间
    with tf.name_scope(layer_name):
        # 调用之前的方法初始化权重w，并且调用参数信息的记录方法，记录w的信息
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim], None)
            #variable_summaries(weights)
        # 调用之前的方法初始化权重b，并且调用参数信息的记录方法，记录b的信息
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            #variable_summaries(biases)
        # 执行wx+b的线性计算，并且用直方图记录下来
        with tf.name_scope('linear_compute'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('linear', preactivate)
        # 将线性输出经过激励函数，并将输出也用直方图记录下来
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
    # 返回激励层的最终输出
    return activations

hidden1 = nn_layer(x, 784, 500, 'layer1')

# Create dropout layer
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)


y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

# 创建损失函数
with tf.name_scope('loss'):
    # 计算交叉熵损失（每个样本都会有一个损失）
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
        # 计算所有样本交叉熵损失的均值
        #cross_entropy = tf.reduce_mean(diff) + tf.add_n(tf.get_collection('losses'))
    tf.summary.scalar('loss', cross_entropy)


# 使用AdamOptimizer优化器训练模型，最小化交叉熵损失
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# 计算准确率
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 分别将预测和真实的标签中取出最大值的索引，弱相同则返回1(true),不同则返回0(false)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        # 求均值即为准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


tf.summary.scalar('accuracy', accuracy)

# summaries合并
merged = tf.summary.merge_all()
# 写到指定的磁盘路径中
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/test')

# 运行初始化所有变量
tf.global_variables_initializer().run()

def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
        xs, ys = mnist.train.next_batch(100)
        k = dropout
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}



saver = tf.train.Saver()
#saver.restore(sess, Save_dir + "/HandWrite")


for i in range(max_steps):
    if i % 10 == 0:  # 记录测试集的summary与accuracy
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        test_writer.add_summary(summary, i)
        saver.save(sess, Save_dir + "/HandWrite")
        print('Accuracy at step %s: %s' % (i, acc))
    else:  # 记录训练集的summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)

train_writer.close()
test_writer.close()
