import math
from vgg16 import Vgg16Model
from data_preprocessing import BatchGenerator
import os
import tensorflow as tf
import  numpy as np
batch_size=4
image_directory = 'data/nyu_datasets'
VGG_MEAN = [103.939, 116.779, 123.68]

def load_image(image_examples):
    images = []
    for image_example in image_examples:
        jpg = tf.read_file(image_example)
        image = tf.image.decode_jpeg(jpg, channels=3)
        image = vgg16_preprocess(image)
        images.append(image)
    return np.array(images)

def vgg16_preprocess(image, shape=(224,224), mean=VGG_MEAN):
    image = tf.image.convert_image_dtype(image, tf.uint8)
    image = tf.reverse(image, axis=[-1])  # RGB to BGR

    image = tf.cast(image, dtype=tf.float32)
    image = tf.subtract(image, mean)

    tf.image.resize_images(image,shape)
    return image

def extractFetures():
    '''
    filenames = os.listdir(image_directory)
    images_filenames = []
    for filename in filenames:
        if filename.endswith('.jpg'):
            images_filenames.append(filename)

    num_files = len(images_filenames)
    num_batches = int(math.ceil(num_files / batch_size))
    '''

    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            batch_generator = BatchGenerator(batch_size=batch_size)
            # train_images, train_depths, train_pixels_mask = batch_generator.csv_inputs(TRAIN_FILE)
            train_images, train_depths, train_pixels_mask, batch_filenames = batch_generator.csv_inputs('train.csv')

        input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name="images")
        print('build')
        vgg = Vgg16Model()
        vgg.build(input)
        print('built')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # initialize the queue threads to start to shovel data
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for i in range(1):
                images,names = sess.run([train_images,batch_filenames])
                batch_features = sess.run(
                    vgg.conv4_1,
                    feed_dict={input: images}
                )

                for i, filename in enumerate(names):
                    np.save(os.path.splitext(os.path.split(filename)[1])[0].decode()+'.npy',batch_features[i])
            # stop our queue threads and properly close the session
            coord.request_stop()
            coord.join(threads)
            sess.close()

def main(argv=None):
    extractFetures()

if __name__ == '__main__':
    tf.app.run()