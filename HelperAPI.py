import tensorflow as tf

# TODO : add layer and parameters name parameter
def weights_init(shape,layer_name,trainable = True):
    '''
    This function is used when weights are initialized.

    Input: shape - list of int numbers which are representing dimensions of our weights.
    '''
    return tf.Variable(tf.truncated_normal(shape, stddev=0.001),name=layer_name+"_B",trainable=trainable)


def bias_init(shape,layer_name,trainable=True):
    '''
    This function is used when biases are initialized.

    Input: shape - scalar that represents length of bias vector for particular layer in a network.
    '''
    return tf.Variable(tf.constant(0.05, shape=shape),name=layer_name+"_W",trainable=trainable)


def conv2d(input, filter_size, number_of_channels, number_of_filters, strides=(1, 1), padding='SAME',
                  activation=tf.nn.relu, max_pool=True,batch_norm=True,layer_name ='',trainable=True):
    '''
    This function is used to create single convolution layer in a CNN network.

    Inputs: input
            filter_size - int value that represents width and height for kernel used in this layer.
            number_of_channels - number of channels that INPUT to this layer has.
            number_of_filters - how many filters in our output do we want, this is going to be number of channels of this layer
                                and this number is used as a number of channels for the next layer.
            strides - how many pixels filter/kernel is going to move per time.
            paddign - if its needed we pad image with zeros. "SAME" = output has same dimensions as an input, "VALID" - this is
                      another option for padding parameter.
            activation - which activation/if any this layer will use
            max_pool - if True output height and width will be half sized the input size.
    '''

    weights = weights_init([filter_size, filter_size, number_of_channels, number_of_filters],layer_name=layer_name,trainable=trainable)
    biases = bias_init([number_of_filters],layer_name=layer_name,trainable=trainable)

    layer = tf.nn.conv2d(input, filter=weights, strides=[1, strides[0], strides[1], 1], padding=padding,name=layer_name+'_conv') + biases
    if batch_norm:
        layer = tf.layers.batch_normalization(layer)
    layer = activation(layer)

    if max_pool:
        layer = tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name=layer_name+'_pool')

    return layer


def flatten(layer):
    '''
    This function should be used AFTER last conv layer in a network.

    This function will take LAYER as an input and output flattend layer. This should be done so we can use fc layer afterwards.
    '''
    shape = layer.get_shape()
    num_of_elements = shape[1:4].num_elements()
    reshaped = tf.reshape(layer, [-1, num_of_elements])
    return reshaped, num_of_elements


def fully_connected(input, input_shape, output_shape, activation=tf.nn.relu, dropout=None,layer_name = '',trainable=True):
    '''
    This function is used to create single fully connected layer in a network.

    Inputs: input
            intput_shape - number of "neurons" of the input to this layer
            output_shape - number of "neurons" that we want to have in this layer
            activation - which activation/if any this layer will use
            dropout - if this is NOT None but some number, we are going to, randomly, turn off neurons in this layer.
    '''

    weights = weights_init([input_shape, output_shape],layer_name=layer_name,trainable=trainable)
    biases = bias_init([output_shape],layer_name=layer_name,trainable=trainable)

    layer = tf.matmul(input, weights) + biases

    if activation != None:
        layer = activation(layer)

    if dropout != None:
        layer = tf.nn.dropout(layer, dropout)

    return layer