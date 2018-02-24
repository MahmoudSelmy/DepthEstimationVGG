import tensorflow as tf
from math import pow
output_size = 24 * 24

def build_loss(scale2_op, depths, pixels_mask):
    # print(pixels_mask.get_shape())
    predictions_all = tf.reshape(scale2_op, [-1, output_size])
    depths_all = tf.reshape(depths, [-1, output_size])
    pixels_mask = tf.reshape(pixels_mask, [-1, output_size])

    # print(predictions_all.get_shape())
    # print(pixels_mask.get_shape())

    # n =tf.reduce_sum(pixels_mask,1) # all_subset_data images does not have any invalid pixels

    # print("n")
    # print(n.get_shape())

    predictions_valid = tf.multiply(predictions_all, pixels_mask)
    target_valid = tf.multiply(depths_all, pixels_mask)

    # print(predictions_valid.get_shape())
    # print(target_valid.get_shape())

    d = tf.subtract(predictions_valid, target_valid)
    square_d = tf.square(d)

    sum_square_d = tf.reduce_sum(square_d, 1)
    # print(sum_square_d.get_shape())

    sum_d = tf.reduce_sum(d, 1)
    # print(sum_square_d.get_shape())

    sqare_sum_d = tf.square(sum_d)

    cost = tf.reduce_mean( (sum_square_d / output_size ) - 0.5* (sqare_sum_d / pow(output_size,2) ))
    # cost = tf.reduce_mean(sum_square_d/output_size)
    cost = tf.sqrt(cost)
    return cost