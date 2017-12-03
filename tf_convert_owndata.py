import tensorflow as tf

from datasets import susimple_lane_to_tfrecords 

def main(_):

    # kitti_to_tfrecords.run('./kitti_data/', './tf_records', 'kitti_train', shuffling = True)

    susimple_lane_to_tfrecords.run('./train_set/', './tf_records',
                                        'tusimple_lane', shuffling = True)
if __name__ == '__main__':
    tf.app.run()
