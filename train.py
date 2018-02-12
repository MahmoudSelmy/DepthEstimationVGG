from datetime import datetime
from tensorflow.python.platform import gfile
import tensorflow as tf
from data_preprocessing import BatchGenerator
from DepthLoss import build_loss
from vgg16 import Vgg16Model
from Utills import output_predict,output_groundtruth

BATCH_SIZE = 4
TRAIN_FILE = "sub_train.csv"
TEST_FILE = "train.csv"
EPOCHS = 2000

IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
TARGET_HEIGHT = 55
TARGET_WIDTH = 74


INITIAL_LEARNING_RATE = 0.0001
LEARNING_RATE_DECAY_FACTOR = 0.9
MOVING_AVERAGE_DECAY = 0.999999
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 500
NUM_EPOCHS_PER_DECAY = 30

SCALE1_DIR = 'Scale1'
SCALE2_DIR = 'Scale2'
logs_path = "/tmp/multi_scale/2"

def train_model():

    # directors to save the chkpts of each scale
    if not gfile.Exists(SCALE1_DIR):
        gfile.MakeDirs(SCALE1_DIR)
    if not gfile.Exists(SCALE2_DIR):
        gfile.MakeDirs(SCALE2_DIR)

    with tf.Graph().as_default():
        # get batch
        global_step = tf.Variable(0, name='global_step', trainable=False)
        with tf.device('/cpu:0'):
            batch_generator = BatchGenerator(batch_size=BATCH_SIZE)
            # train_images, train_depths, train_pixels_mask = batch_generator.csv_inputs(TRAIN_FILE)
            train_images, train_depths, train_pixels_mask,names = batch_generator.csv_inputs(TRAIN_FILE)
        '''
        # placeholders
            training_images = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name="training_images")
            depths = tf.placeholder(tf.float32, shape=[None, TARGET_HEIGHT, TARGET_WIDTH, 1], name="depths")
            pixels_mask = tf.placeholder(tf.float32, shape=[None, TARGET_HEIGHT, TARGET_WIDTH, 1], name="pixels_mask")
        '''

        # build model
        vgg = Vgg16Model()
        vgg.build(train_images)

        loss = build_loss(scale2_op=vgg.outputdepth, depths=train_depths, pixels_mask=train_pixels_mask)

        #learning rate
        num_batches_per_epoch = float(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN) / BATCH_SIZE
        '''
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        lr = tf.train.exponential_decay(
            INITIAL_LEARNING_RATE,
            global_step,
            100000,
            LEARNING_RATE_DECAY_FACTOR,
            staircase=True)
        '''


        #optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=INITIAL_LEARNING_RATE).minimize(loss, global_step=global_step)
        # TODO: define model saver

        # Training session
        # sess_config = tf.ConfigProto(log_device_placement=True)
        # sess_config.gpu_options.allocator_type = 'BFC'
        # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.80

        '''
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        '''

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # initialize the queue threads to start to shovel data
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # p = tf.Print(data_file,[data_file],message=)

            for epoch in range(EPOCHS):
                for i in range(1000):
                    _, loss_value, out_depth, ground_truth, batch_images = sess.run(
                        [optimizer, loss, vgg.outputdepth, train_depths, train_images])

                    # validation_loss, _ = sess.run([loss, train_images])


                    if i % 50 == 0:
                        # log.info('step' + loss_value)
                        print("%s: %d[epoch]: %d[iteration]: train loss %f " % (datetime.now(), epoch, i, loss_value))

                    # print("%s: %d[epoch]: %d[iteration]: train loss %f" % (datetime.now(), epoch, i, loss_value))
                    if i % 100 == 0:
                        output_groundtruth(out_depth, ground_truth,"data/predictions/predict_scale1_%05d_%05d" % (epoch, i))

            # stop our queue threads and properly close the session
            coord.request_stop()
            coord.join(threads)
            sess.close()

def main(argv=None):
    train_model()

if __name__ == '__main__':
    tf.app.run()