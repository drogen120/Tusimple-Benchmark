import json
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
from nets import nets_factory
import tf_utils

slim = tf.contrib.slim
BATCH_SIZE = 8
DATA_FORMAT = 'NHWC'
# =========================================================================== #
# Fine-Tuning Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'checkpoint_path', './checkpoints/mobilenet_v1_1.0_224.ckpt',
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', None,
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')
tf.app.flags.DEFINE_integer('max_number_of_steps', 9000000,
                           'The maximum number of training steps')

# =========================================================================== #
# Optimization Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')
tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')
tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float(
    'opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')
tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.96, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 20.0,
    'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')
tf.app.flags.DEFINE_integer(
    'batch_size', 8,
    'traing batch size'
)
FLAGS = tf.app.flags.FLAGS

def _parse_function(example_proto):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/line1': tf.FixedLenFeature([56], dtype=tf.float32),
        'image/line2': tf.FixedLenFeature([56], dtype=tf.float32),
        'image/line3': tf.FixedLenFeature([56], dtype=tf.float32),
        'image/line4': tf.FixedLenFeature([56], dtype=tf.float32),
        'image/line5': tf.FixedLenFeature([56], dtype=tf.float32),
        'image/ysamples': tf.FixedLenFeature([56], dtype=tf.float32),
    }
    parsed_features = tf.parse_single_example(example_proto, keys_to_features)
    image_decoded = tf.image.decode_image(parsed_features['image/encoded'])
    
    line1 = parsed_features['image/line1']
    line2 = parsed_features['image/line2']
    line3 = parsed_features['image/line3']
    line4 = parsed_features['image/line4']
    line5 = parsed_features['image/line5']
    ysamples = parsed_features['image/ysamples']
    shape = parsed_features['image/shape']
    width = parsed_features['image/width']
    height = parsed_features['image/height']
    channels= parsed_features['image/channels']
    # print height.get_shape().as_list()
    # image_decoded = tf.reshape(image_decoded, shape) 
    image_decoded.set_shape([720, 1280, 3])
    return image_decoded, [line1, line2, line3, line4, line5], ysamples, shape

def update_gt_map(x_index, y_index, gt_map):
    print tf.shape(x_index)
    upper = y_index - 1 
    lower = 56 - y_index
    left = x_index - 1 
    right = 56 - x_index
    # one_value = tf.constant([[0,0,0],[0,1,0],[0,0,0]])
    one_value = tf.ones([1, 1], tf.float32)
    paddings = tf.cast(tf.stack([upper, lower, left, right]), tf.int32)
    paddings = tf.reshape(paddings, (2,2))
    new_gt = tf.pad(one_value, paddings, "CONSTANT")
    gt_map = tf.add(gt_map, new_gt)
    gt_map = tf.minimum(gt_map, 1)
    return gt_map

def copy_gt_map(gt_map):
    return gt_map

def add_line_num(line_num, index):
    index = 0
    return line_num+1, index

def copy_line_num(line_num, index):
    return line_num, index

def gt_cond(line_num, index, x_gred, y_gred, gt_map):
    print line_num
    return line_num < 5

def gt_body(line_num, index, x_gred, y_gred, gt_map):
    x_index = tf.floor(x_gred[line_num, index])
    y_index = tf.floor(y_gred[line_num, index])
    gt_map = tf.cond(x_index > 0 , lambda: update_gt_map(x_index, y_index, gt_map), 
                     lambda: copy_gt_map(gt_map))

    index += 1

    line_num, index = tf.cond(index > 55, 
                              lambda: add_line_num(line_num, index),
                              lambda: copy_line_num(line_num, index))

    print line_num, index
    return line_num, index, x_gred, y_gred, gt_map


def gt_encoder(training_element, gred_size=56):
    img, lines, ysamples, img_shape = training_element
    ysamples = tf.stack([ysamples] * 5, axis=1)
    height, width, _ = tf.split(img_shape, 3, axis=1)
    height = tf.stack([height] * 5, axis=1)
    width = tf.stack([width] * 5, axis=1)
    x_greds = tf.div(lines, tf.cast(width, tf.float32))
    y_greds = tf.div(ysamples, tf.cast(height, tf.float32))
    x_greds = x_greds * gred_size
    y_greds = y_greds * gred_size
    gt_maps = []
    images_resized = []
    for i in range(BATCH_SIZE):
        # image_reshape = tf.reshape(img[i], img_shape[i])
        # print image_reshape.get_shape().as_list()
        image_resize = tf.image.resize_images(img[i], [448, 448])
        x_gred = x_greds[i, :, :]
        y_gred = y_greds[i, :, :]
        line_num = 0
        index = 0
        gt_map = tf.zeros([56, 56], tf.float32)
        result = tf.while_loop(gt_cond, gt_body, [line_num, index, x_gred, y_gred,
                                            gt_map])
        gt_maps.append(result[4])
        images_resized.append(image_resize)

    images_resized = tf.stack(images_resized)
    gt_maps = tf.stack(gt_maps)

    return images_resized, gt_maps, img_shape

def main(_):
    tf.logging.set_verbosity(tf.logging.DEBUG)

    with tf.Graph().as_default():

        tf_recorders = glob.glob('./tf_records/*.tfrecord')
        sess = tf.InteractiveSession()
        dataset = tf.data.TFRecordDataset(tf_recorders)
        global_step = slim.create_global_step()
        dataset = dataset.map(_parse_function)
        dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.repeat(10)
        dataset = dataset.batch(BATCH_SIZE)
        print dataset.output_shapes
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        img, gt_maps, img_shape = gt_encoder(next_element)
        print img.get_shape().as_list()
        print tf.shape(gt_maps)
        print tf.shape(img_shape)
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        summaries.add(tf.summary.image("input_image", tf.cast(img, tf.float32)))
        summaries.add(tf.summary.image("gt_map", tf.cast(tf.expand_dims(gt_maps, -1), tf.float32)))
       
        net_class = nets_factory.get_network('mobilenet_lane_net')
        net_params = net_class.default_params._replace(num_classes=1)
        lane_net = net_class(net_params)
        net_shape = lane_net.params.img_shape

        arg_scope = lane_net.arg_scope(weight_decay=0.00004,
                                       data_format=DATA_FORMAT)

        with slim.arg_scope(arg_scope):
            lane_prediction, lane_logits, end_points = \
                    lane_net.net(img, is_training = True)

        lane_net.losses(lane_logits, gt_maps, 0)
        total_loss = tf.losses.get_total_loss()
        summaries.add(tf.summary.scalar('loss', total_loss))

        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            summaries.add(tf.summary.scalar(loss.op.name, loss))

        for variable in tf.trainable_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        with tf.name_scope('Optimizer'):
            learning_rate = tf_utils.configure_learning_rate(
                FLAGS, 700, global_step
            ) 
            optimizer = tf_utils.configure_optimizer(FLAGS, learning_rate)

        summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = slim.learning.create_train_op(
                total_loss,
                optimizer,
                summarize_gradients=False
            )
            
        summary_op = tf.summary.merge(list(summaries), name='summary_op')
        train_writer = tf.summary.FileWriter('./logs/', sess.graph)
        # variables_to_train = tf.trainable_variables()
        # variables_to_restore = \
                # tf.contrib.framework.filter_variables(variables_to_train,
                                                     # exclude_patterns=['_box',
                                                                      # '_fpn'])
        # restorer = tf.train.Saver(variables_to_restore)
        saver = tf.train.Saver(max_to_keep=5,
                              keep_checkpoint_every_n_hours=1.0,
                              write_version=2,
                              pad_step_number=False)

        sess.run(tf.global_variables_initializer())
        # restorer.restore(sess, FLAGS.checkpoint_path)

        i = 0
        with slim.queues.QueueRunners(sess):
            sess.run(iterator.initializer)
            while (i < FLAGS.max_number_of_steps):
                _, summary_str = sess.run([train_op, summary_op])
                if i % 50 == 0:
                    global_step_str = global_step.eval()
                    print('%d iteration' % (global_step_str))
                    train_writer.add_summary(summary_str, global_step_str)
                if i % 100 == 0:
                    global_step_str = global_step.eval()
                    saver.save(sess, "./logs/", global_step=global_step_str)

                i += 1

        # with slim.queues.QueueRunners(sess):
            # i = 0
            # sess.run(iterator.initializer)
            # while (i < 3):
                # _, summary_str = sess.run([gt_maps, summary_op])
                # global_step_str = global_step.eval()
                # train_writer.add_summary(summary_str, global_step_str)
                # i += 1
                # print ('====================')

if __name__ == '__main__':
    tf.app.run()

