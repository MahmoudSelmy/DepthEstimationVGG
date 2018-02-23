import tensorflow as tf
import numpy as np
import HelperAPI as helper

output_size = 30 * 30

class Vgg16Model:
    def __init__(self, weights_path='./vgg16.npy'):
        self.weights = np.load('vgg16.npy', encoding='latin1').item()
        self.activation_fn = tf.nn.relu
        self.conv_padding = 'SAME'
        self.pool_padding = 'SAME'
        self.use_bias = True

    def build(self, input_tensor, trainable=False,isTraining=True):
        self.conv1_1 = self.conv2d(input_tensor, 'conv1_1', 64, trainable)
        self.conv1_2 = self.conv2d(self.conv1_1, 'conv1_2', 64, trainable)

        # Max-pooling is performed over a 2 × 2 pixel window, with stride 2.
        self.max_pool1 = tf.layers.max_pooling2d(self.conv1_2, (2, 2), (2, 2), padding=self.pool_padding)

        self.conv2_1 = self.conv2d(self.max_pool1, 'conv2_1', 128, trainable)
        self.conv2_2 = self.conv2d(self.conv2_1, 'conv2_2', 128, trainable)

        self.max_pool2 = tf.layers.max_pooling2d(self.conv2_2, (2, 2), (2, 2), padding=self.pool_padding)

        self.conv3_1 = self.conv2d(self.max_pool2, 'conv3_1', 256, trainable)
        self.conv3_2 = self.conv2d(self.conv3_1, 'conv3_2', 256, trainable)
        self.conv3_3 = self.conv2d(self.conv3_2, 'conv3_3', 256, trainable)

        self.max_pool3 = tf.layers.max_pooling2d(self.conv3_3, (2, 2), (2, 2), padding=self.pool_padding)

        self.conv4_1 = self.conv2d(self.max_pool3, 'conv4_1', 512, trainable)

        self.conv4_2 = self.conv2d(self.conv4_1, 'conv4_2', 512, trainable)
        self.conv4_3 = self.conv2d(self.conv4_2, 'conv4_3', 512, trainable)

        self.max_pool4 = tf.layers.max_pooling2d(self.conv4_3, (2, 2), (2, 2), padding=self.pool_padding)

        self.conv5_1 = self.conv2d(self.max_pool4, 'conv5_1', 512, trainable= True)
        # self.conv5_1 = self.conv2d(self.max_pool4, 'conv5_1', n_channel= 512 ,n_filters=512, reuse =False)
        # retrain
        self.conv5_2 = self.conv2d(self.conv5_1, 'conv5_2', n_channel= 512 ,n_filters=512,isTraining=isTraining)
        self.conv5_3 = self.conv2d(self.conv5_2, 'conv5_3', n_channel= 512 ,n_filters=512, reuse = False,isTraining=isTraining)

        self.max_pool5 = tf.layers.max_pooling2d(self.conv5_3, (2, 2), (2, 2), padding=self.pool_padding)
        '''
        conv_6_s1 = helper.conv2d(input=self.max_pool5, filter_size=1, number_of_channels=512, number_of_filters=128,
                                  padding='VALID',
                                  max_pool=False, layer_name='conv_6_s1',isTraining=isTraining)
        '''
        reshaped = tf.reshape(self.max_pool5, shape=(-1, 7 * 7 * 512))
        shape = reshaped.get_shape()
        n_elements = shape[1:4].num_elements()

        self.fc6 = self.fc(reshaped, 'fc6', size=4096,input_size=n_elements,reuse =False,isTraining=isTraining,dropout=0.5)
        self.fc7 = self.fc(self.fc6, 'fc7', size=4096,input_size=4096, reuse =False,isTraining=isTraining,dropout=0.5)

        self.fc8 = self.fc(self.fc7, 'fc8', size=output_size,input_size=4096, reuse =False,isTraining=isTraining,batch_norm=False)
        self.outputdepth = tf.reshape(self.fc8, [-1, 30, 30, 1])




    def conv2d(self, layer, name, n_filters, trainable=True, k_size=3,reuse = True,n_channel=3,isTraining=True):
        if reuse:
            layer = tf.layers.conv2d(layer, n_filters, kernel_size=(k_size, k_size),
                                activation=self.activation_fn, padding=self.conv_padding, name=name, trainable=trainable,
                                kernel_initializer=tf.constant_initializer(self.weights[name][0], dtype=tf.float32),
                                bias_initializer=tf.constant_initializer(self.weights[name][1], dtype=tf.float32),
                                use_bias=self.use_bias)
        else :
            layer = helper.conv2d(input=layer, filter_size=k_size, number_of_channels=n_channel, number_of_filters=n_filters,
                              padding=self.conv_padding,
                              max_pool=False,layer_name=name,batch_norm=True,isTraining=isTraining)
        return layer

    def fc(self, layer, name, size, trainable=True,reuse=True,input_size = 1024,isTraining=True,dropout=None,batch_norm=True):
        if reuse :
            layer = tf.layers.dense(layer, size, activation=self.activation_fn,
                                    name=name, trainable=trainable,
                                    kernel_initializer=tf.constant_initializer(self.weights[name][0], dtype=tf.float32),
                                    bias_initializer=tf.constant_initializer(self.weights[name][1], dtype=tf.float32),
                                    use_bias=self.use_bias)
        else:
            layer = helper.fully_connected(input=layer, input_shape=input_size, output_shape=size, layer_name=name,isTraining=isTraining,dropout=dropout,batch_norm=batch_norm)
        return layer