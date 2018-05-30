#encoding:utf-8

import PIL.Image
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os


def imagePrepare(PicDir):
    img = PIL.Image.open(PicDir)
    img_reset = img.resize((28,28))
    Lim  =  img_reset.convert("L")
    Lim.show()
    threshold  =   125
    table  =  []
    for  i  in  range( 256 ):
         if  i  <  threshold:
            table.append(0)
         else :
            table.append( 1 )
    bim  =  Lim.point(table,  "1")

    tv = list(bim.convert("L").getdata())
    tva = [ (255-x)*1.0/255.0 for x in tv]
    #print len(tva)
    return tva



max_steps = 20000
learning_rate = 1e-4
REGULARIZATION_RATE = 0.0001
dropout = 0.9
data_dir = './MNIST_data'
log_dir = './MNIST_logs_CNN'
Save_dir = './MNIST_Model_CNN'

#default regularizer
regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

# get the data
#mnist = input_data.read_data_sets(data_dir,one_hot = True)

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

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)


def nn_layer_CNN(input_tensor, act=tf.nn.relu):
    # convolution one
    with tf.name_scope('convolution_one'):
        with tf.name_scope('weights'):
            weights_one = weight_variable([5,5,1,32], regularizer)
        with tf.name_scope('biases'):
            bias_one = bias_variable([32])
            output_convolution1 = max_pool_2x2(act(conv2d(input_tensor, weights_one) + bias_one))

    # convolution two
    with tf.name_scope('convolution_two'):
        with tf.name_scope('weights'):
            weights_two = weight_variable([5,5,32,64], regularizer)
        with tf.name_scope('biases'):
            bias_two = bias_variable([64])
            output_convolution2 = max_pool_2x2(act(conv2d(output_convolution1, weights_two) + bias_two))
    # full_connect
    with tf.name_scope('full_connect'):
        input_FC = tf.reshape(output_convolution2,[-1, 7*7*64])
        with tf.name_scope('weights'):
            weights_FC = weight_variable([7*7*64, 1024], regularizer)
        with tf.name_scope('biases'):
            bias_FC = bias_variable([1024])
            output_FC = act(tf.matmul(input_FC, weights_FC) + bias_FC)
    # output
    with tf.name_scope('Final_Output'):
        # add the dropout\
        #tf.summary.scalar('dropout_keep_probability', keep_prob)
        output_FC_dropout = tf.nn.dropout(output_FC, keep_prob)
        with tf.name_scope('weights'):
            weights_FO = weight_variable([1024, 10], regularizer)
        with tf.name_scope('biases'):
            bias_FO = bias_variable([10])
            output_FO = tf.nn.softmax(tf.matmul(output_FC_dropout, weights_FO) + bias_FO)

    return output_FO

y = nn_layer_CNN(image_shaped_input);


# 创建损失函数
with tf.name_scope('loss'):
    # 计算交叉熵损失（每个样本都会有一个损失）
    cross_entropy = -tf.reduce_sum(y_*tf.log(y)) + tf.add_n(tf.get_collection('losses'))

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


# 运行初始化所有变量
tf.global_variables_initializer().run()

saver = tf.train.Saver()
saver.restore(sess, Save_dir + "/HandWrite")

ImageDir = "./Test.jpeg"
img = imagePrepare(ImageDir)

prediction=tf.argmax(y,1)
predint=prediction.eval(feed_dict={x: [img],keep_prob: 1.0}, session=sess)

print('recognize result:')
print(predint[0])