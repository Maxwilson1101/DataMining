# Chapter 14 实现复杂CNN
import os
import struct
import numpy as np
import tensorflow as tf


# 自定义独热编码函数
def dense_to_one_hot(labels_dense, num_classes=10):
    '''将类标签从标量转换为一个独热向量'''
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# 自定义下载函数
def load_mnist(path, kind='train'):
    '''根据指定路径加载数据集'''
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
        labels = dense_to_one_hot(labels)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


# 定义网络参数
# 参数
learning_rate = 0.001
num_steps = 4000
batch_size = 128
display_step = 100
# 网络参数
n_input = 784
n_classes = 10
dropout = 0.80


# 导入数据
x_train, y_train = load_mnist('./MNIST_data', kind='train')
print('Rows: %d, columns: %d' % (x_train.shape[0], x_train.shape[1]))
print('Rows: %d, columns: %d' % (y_train.shape[0], y_train.shape[1]))

x_test, y_test = load_mnist('./MNIST_data/', kind='t10k')
print('Rows: %d, columns: %d' % (x_test.shape[0], x_test.shape[1]))


# 创建Dataset数据集
sess = tf.Session()

# 创建来自images and the labels的数据集张量
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train.astype(np.float32), y_train.astype(np.float32))
)
# 对数据集进行批次划分
dataset = dataset.batch(batch_size)
# 在dataset创建迭代器
iterator = dataset.make_initializable_iterator()
# 使用两个展位符号(placeholders), 避免2G限制
_data = tf.placeholder(tf.float32, [None, n_input])
_labels = tf.placeholder(tf.float32, [None, n_classes])
# 初始化迭代器
sess.run(iterator.initializer, feed_dict={_data: x_train.astype(np.float32),
                                          _labels: y_train.astype(np.float32)})
# 获取输入值
X, Y = iterator.get_next()


# 创建模型
def conv_net(x, n_classes, dropout, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse=reuse):

