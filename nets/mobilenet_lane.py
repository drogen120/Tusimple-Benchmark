# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""MobileNet v1.

MobileNet is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and different
head (for example: embeddings, localization and classification).

As described in https://arxiv.org/abs/1704.04861.

  MobileNets: Efficient Convolutional Neural Networks for
    Mobile Vision Applications
  Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang,
    Tobias Weyand, Marco Andreetto, Hartwig Adam

100% Mobilenet V1 (base) with input size 224x224:

Layer                                                     params           macs
--------------------------------------------------------------------------------
MobilenetV1/Conv2d_0/Conv2D:                                 864      10,838,016
MobilenetV1/Conv2d_1_depthwise/depthwise:                    288       3,612,672
MobilenetV1/Conv2d_1_pointwise/Conv2D:                     2,048      25,690,112
MobilenetV1/Conv2d_2_depthwise/depthwise:                    576       1,806,336
MobilenetV1/Conv2d_2_pointwise/Conv2D:                     8,192      25,690,112
MobilenetV1/Conv2d_3_depthwise/depthwise:                  1,152       3,612,672
MobilenetV1/Conv2d_3_pointwise/Conv2D:                    16,384      51,380,224
MobilenetV1/Conv2d_4_depthwise/depthwise:                  1,152         903,168
MobilenetV1/Conv2d_4_pointwise/Conv2D:                    32,768      25,690,112
MobilenetV1/Conv2d_5_depthwise/depthwise:                  2,304       1,806,336
MobilenetV1/Conv2d_5_pointwise/Conv2D:                    65,536      51,380,224
MobilenetV1/Conv2d_6_depthwise/depthwise:                  2,304         451,584
MobilenetV1/Conv2d_6_pointwise/Conv2D:                   131,072      25,690,112
MobilenetV1/Conv2d_7_depthwise/depthwise:                  4,608         903,168
MobilenetV1/Conv2d_7_pointwise/Conv2D:                   262,144      51,380,224
MobilenetV1/Conv2d_8_depthwise/depthwise:                  4,608         903,168
MobilenetV1/Conv2d_8_pointwise/Conv2D:                   262,144      51,380,224
MobilenetV1/Conv2d_9_depthwise/depthwise:                  4,608         903,168
MobilenetV1/Conv2d_9_pointwise/Conv2D:                   262,144      51,380,224
MobilenetV1/Conv2d_10_depthwise/depthwise:                 4,608         903,168
MobilenetV1/Conv2d_10_pointwise/Conv2D:                  262,144      51,380,224
MobilenetV1/Conv2d_11_depthwise/depthwise:                 4,608         903,168
MobilenetV1/Conv2d_11_pointwise/Conv2D:                  262,144      51,380,224
MobilenetV1/Conv2d_12_depthwise/depthwise:                 4,608         225,792
MobilenetV1/Conv2d_12_pointwise/Conv2D:                  524,288      25,690,112
MobilenetV1/Conv2d_13_depthwise/depthwise:                 9,216         451,584
MobilenetV1/Conv2d_13_pointwise/Conv2D:                1,048,576      51,380,224
--------------------------------------------------------------------------------
Total:                                                 3,185,088     567,716,352


75% Mobilenet V1 (base) with input size 128x128:

Layer                                                     params           macs
--------------------------------------------------------------------------------
MobilenetV1/Conv2d_0/Conv2D:                                 648       2,654,208
MobilenetV1/Conv2d_1_depthwise/depthwise:                    216         884,736
MobilenetV1/Conv2d_1_pointwise/Conv2D:                     1,152       4,718,592
MobilenetV1/Conv2d_2_depthwise/depthwise:                    432         442,368
MobilenetV1/Conv2d_2_pointwise/Conv2D:                     4,608       4,718,592
MobilenetV1/Conv2d_3_depthwise/depthwise:                    864         884,736
MobilenetV1/Conv2d_3_pointwise/Conv2D:                     9,216       9,437,184
MobilenetV1/Conv2d_4_depthwise/depthwise:                    864         221,184
MobilenetV1/Conv2d_4_pointwise/Conv2D:                    18,432       4,718,592
MobilenetV1/Conv2d_5_depthwise/depthwise:                  1,728         442,368
MobilenetV1/Conv2d_5_pointwise/Conv2D:                    36,864       9,437,184
MobilenetV1/Conv2d_6_depthwise/depthwise:                  1,728         110,592
MobilenetV1/Conv2d_6_pointwise/Conv2D:                    73,728       4,718,592
MobilenetV1/Conv2d_7_depthwise/depthwise:                  3,456         221,184
MobilenetV1/Conv2d_7_pointwise/Conv2D:                   147,456       9,437,184
MobilenetV1/Conv2d_8_depthwise/depthwise:                  3,456         221,184
MobilenetV1/Conv2d_8_pointwise/Conv2D:                   147,456       9,437,184
MobilenetV1/Conv2d_9_depthwise/depthwise:                  3,456         221,184
MobilenetV1/Conv2d_9_pointwise/Conv2D:                   147,456       9,437,184
MobilenetV1/Conv2d_10_depthwise/depthwise:                 3,456         221,184
MobilenetV1/Conv2d_10_pointwise/Conv2D:                  147,456       9,437,184
MobilenetV1/Conv2d_11_depthwise/depthwise:                 3,456         221,184
MobilenetV1/Conv2d_11_pointwise/Conv2D:                  147,456       9,437,184
MobilenetV1/Conv2d_12_depthwise/depthwise:                 3,456          55,296
MobilenetV1/Conv2d_12_pointwise/Conv2D:                  294,912       4,718,592
MobilenetV1/Conv2d_13_depthwise/depthwise:                 6,912         110,592
MobilenetV1/Conv2d_13_pointwise/Conv2D:                  589,824       9,437,184
--------------------------------------------------------------------------------
Total:                                                 1,800,144     106,002,432

"""

# Tensorflow mandates these.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import tf_extended as tfe
from nets import custom_layers
from nets import ssd_common

import tensorflow as tf
import numpy as np
import math

slim = tf.contrib.slim

LaneParams = namedtuple('LaneParams', ['img_shape',
                                       'num_classes',
                                       'feat_layers',
                                       'feat_shapes'
                                         ])

# Conv and DepthSepConv namedtuple define layers of the MobileNet architecture
# Conv defines 3x3 convolution layers
# DepthSepConv defines 3x3 depthwise convolution followed by 1x1 convolution.
# stride is the stride of the convolution
# depth is the number of channels or filters in a layer
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])

# _CONV_DEFS specifies the MobileNet body
_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=32),  # 224*224 0
    DepthSepConv(kernel=[3, 3], stride=1, depth=64), # 224*224 1
    DepthSepConv(kernel=[3, 3], stride=2, depth=128), # 112*112 2
    DepthSepConv(kernel=[3, 3], stride=1, depth=128), # 112*112 3
    DepthSepConv(kernel=[3, 3], stride=2, depth=256), # 56 * 56 4
    DepthSepConv(kernel=[3, 3], stride=1, depth=256), # 56 * 56 5
    DepthSepConv(kernel=[3, 3], stride=2, depth=512), # 28 * 28 6
    DepthSepConv(kernel=[3, 3], stride=1, depth=512), # 28 * 28 7
    DepthSepConv(kernel=[3, 3], stride=1, depth=512), # 28 * 28 8
    DepthSepConv(kernel=[3, 3], stride=2, depth=512), # 28 * 28 9
    DepthSepConv(kernel=[3, 3], stride=1, depth=512), # 14 * 14 10
    DepthSepConv(kernel=[3, 3], stride=1, depth=512), # 14 * 14 11
    DepthSepConv(kernel=[3, 3], stride=2, depth=1024),# 7 *  7  12
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024) # 7 * 7   13
]
class Mobilenet_Lane_Net(object):
    default_params = LaneParams(
        img_shape=(448, 448),
        num_classes=2,
        feat_layers=['Conv2d_5_pointwise', 'Conv2d_8_pointwise', 'Conv2d_11_pointwise', 'Conv2d_13_pointwise'],
        feat_shapes=[(56, 56), (28, 28), (14, 14), (7, 7)],
        )

    def __init__(self, params=None):
        """Init the SSD net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, LaneParams):
            self.params = params
        else:
            self.params = Mobilenet_Lane_Net.default_params

    def net(self, inputs,
            is_training=True,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='MobilenetV1'):
        """SSD network definition.
        """
        print ("11111111111111111111")
        print (tf.shape(inputs))
        print ("11111111111111111111")
        r = mobilenet_lane_net(inputs,
                    num_classes=self.params.num_classes,
                    feat_layers=self.params.feat_layers,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope)
        return r

    def arg_scope(self, weight_decay=0.0005, data_format='NHWC', is_training= True):
        """Network arg_scope.
        """
        return mobilenet_lane_net_arg_scope(is_training=is_training, weight_decay = weight_decay, data_format=data_format)

    def losses(self, logits, gt_maps,
               part_num,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope='mobile_lane_losses'):
        """Define the SSD network losses.
        """
        if part_num == 0:
            return lane_net_losses(logits, gt_maps,
                              negative_ratio=negative_ratio,
                              alpha=alpha,
                              label_smoothing=label_smoothing,
                              scope=scope)


def _reduced_kernel_size_for_small_input(self, input_tensor, kernel_size):
    """Define kernel size which is automatically reduced for small input.

    If the shape of the input images is unknown at graph construction time this
    function assumes that the input images are large enough.

    Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

    Returns:
    a tensor with the kernel size.
    """
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1])]
    return kernel_size_out

def mobilenet_lane_base(inputs,
                      final_endpoint='Conv2d_13_pointwise',
                      min_depth=8,
                      depth_multiplier=1.0,
                      conv_defs=None,
                      output_stride=None,
                      scope=None):
    """Mobilenet v1.

    Constructs a Mobilenet v1 network from inputs to the given final endpoint.

    Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_0', 'Conv2d_1_pointwise', 'Conv2d_2_pointwise',
      'Conv2d_3_pointwise', 'Conv2d_4_pointwise', 'Conv2d_5'_pointwise,
      'Conv2d_6_pointwise', 'Conv2d_7_pointwise', 'Conv2d_8_pointwise',
      'Conv2d_9_pointwise', 'Conv2d_10_pointwise', 'Conv2d_11_pointwise',
      'Conv2d_12_pointwise', 'Conv2d_13_pointwise'].
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    conv_defs: A list of ConvDef namedtuples specifying the net architecture.
    output_stride: An integer that specifies the requested ratio of input to
      output spatial resolution. If not None, then we invoke atrous convolution
      if necessary to prevent the network from reducing the spatial resolution
      of the activation maps. Allowed values are 8 (accurate fully convolutional
      mode), 16 (fast fully convolutional mode), 32 (classification mode).
    scope: Optional variable_scope.

    Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.

    Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
                or depth_multiplier <= 0, or the target output_stride is not
                allowed.
    """
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    end_points = {}

    # Used to find thinned depths for each layer.
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    if conv_defs is None:
        conv_defs = _CONV_DEFS

    if output_stride is not None and output_stride not in [8, 16, 32]:
        raise ValueError('Only allowed output_stride values are 8, 16, 32.')

    # with tf.variable_scope(scope, 'MobilenetV1', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):
        # The current_stride variable keeps track of the output stride of the
        # activations, i.e., the running product of convolution strides up to the
        # current network layer. This allows us to invoke atrous convolution
        # whenever applying the next convolution would result in the activations
        # having output stride larger than the target output_stride.
        current_stride = 1

        # The atrous convolution rate parameter.
        rate = 1

        net = inputs
        for i, conv_def in enumerate(conv_defs):
            end_point_base = 'Conv2d_%d' % i

            if output_stride is not None and current_stride == output_stride:
                # If we have reached the target output_stride, then we need to employ
                # atrous convolution with stride=1 and multiply the atrous rate by the
                # current unit's stride for use in subsequent layers.
                layer_stride = 1
                layer_rate = rate
                rate *= conv_def.stride
            else:
                layer_stride = conv_def.stride
                layer_rate = 1
                current_stride *= conv_def.stride

            if isinstance(conv_def, Conv):
                end_point = end_point_base
                net = slim.conv2d(net, depth(conv_def.depth), conv_def.kernel,
                                stride=conv_def.stride,
                                normalizer_fn=slim.batch_norm,
                                scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points

            elif isinstance(conv_def, DepthSepConv):
                end_point = end_point_base + '_depthwise'

              # By passing filters=None
              # separable_conv2d produces only a depthwise convolution layer
                net = slim.separable_conv2d(net, None, conv_def.kernel,
                                          depth_multiplier=1,
                                          stride=layer_stride,
                                          rate=layer_rate,
                                          normalizer_fn=slim.batch_norm,
                                          scope=end_point)

                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points

                end_point = end_point_base + '_pointwise'

                net = slim.conv2d(net, depth(conv_def.depth), [1, 1],
                                stride=1,
                                normalizer_fn=slim.batch_norm,
                                scope=end_point)

                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points
            else:
                raise ValueError('Unknown convolution type %s for layer %d'
                           % (conv_def.ltype, i))
    raise ValueError('Unknown final endpoint %s' % final_endpoint)

def mobilenet_lane_net(inputs,
                       num_classes=Mobilenet_Lane_Net.default_params.num_classes,
                       feat_layers=Mobilenet_Lane_Net.default_params.feat_layers,
                       dropout_keep_prob=0.999,
                       is_training=True,
                       min_depth=8,
                       depth_multiplier=1.0,
                       conv_defs=None,
                       prediction_fn=tf.contrib.layers.softmax,
                       spatial_squeeze=True,
                       reuse=None,
                       scope='MobilenetV1'):
    """Mobilenet v1 model for classification.

    Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    dropout_keep_prob: the percentage of activation values that are retained.
    is_training: whether is training or not.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    conv_defs: A list of ConvDef namedtuples specifying the net architecture.
    prediction_fn: a function to get predictions out of logits.
    spatial_squeeze: if True, logits is of shape is [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

    Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, num_classes]
    end_points: a dictionary from components of the network to the corresponding
      activation.

    Raises:
    ValueError: Input rank is invalid.
    """
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                         len(input_shape))

    with tf.variable_scope(scope, 'MobilenetV1', [inputs, num_classes],
                         reuse=reuse) as net_scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            if reuse:
                net_scope.reuse_variables()
            net, end_points = mobilenet_lane_base(inputs, scope=scope,
                                              min_depth=min_depth,
                                              depth_multiplier=depth_multiplier,
                                              conv_defs=conv_defs)

            with tf.variable_scope('FPN'):
                for i in range(len(feat_layers)-2, -1, -1):
                    endpoint_name = feat_layers[i] + '_fpn'
                    output_num = \
                    _, height, width, depth = end_points[feat_layers[i]].get_shape().as_list()
                    with tf.variable_scope(endpoint_name):
                        feature_net = slim.conv2d(end_points[feat_layers[i+1]],
                                          depth, [3, 3])
                        feature_net = tf.image.resize_bilinear(feature_net,
                                                               [height, width])
                        end_points[feat_layers[i]] = \
                        end_points[feat_layers[i]] + feature_net

            with tf.variable_scope('Lane_Prediction'):
                lane_net = \
                custom_layers.non_local_block(end_points[feat_layers[0]])
                lane_net = slim.conv2d(lane_net,
                                             128, [3, 3])
                lane_logits = slim.conv2d(lane_net, num_classes, [3, 3])
                lane_prediction = prediction_fn(lane_logits)
    
            # with tf.variable_scope('Box'):
                # for i, layer in enumerate(feat_layers):
                    # with tf.variable_scope(layer + '_box'):
                        # p, l = ssd_multibox_layer(end_points[layer],
                                                  # num_classes,
                                                  # anchor_sizes[i],
                                                  # anchor_ratios[i],
                                                  # normalizations[i])
                    # predictions.append(prediction_fn(p))
                    # logits.append(p)
                    # localisations.append(l)

    return lane_prediction, lane_logits, end_points

mobilenet_lane_net.default_image_size = 448

def mobilenet_lane_net_arg_scope(is_training=True,
                           weight_decay=0.00004,
                           stddev=0.09,
                           data_format='NHWC',
                           regularize_depthwise=False):
    """Defines the default MobilenetV1 arg scope.

    Args:
    is_training: Whether or not we're training the model.
    weight_decay: The weight decay to use for regularizing the model.
    stddev: The standard deviation of the trunctated normal weight initializer.
    regularize_depthwise: Whether or not apply regularization on depthwise.

    Returns:
    An `arg_scope` to use for the mobilenet v1 model.
    """
    batch_norm_params = {
        'is_training': is_training,
        'center': True,
        'scale': True,
        'decay': 0.99,
        'epsilon': 0.001,
    }

  # Set weight_decay for weights in Conv and DepthSepConv layers.
    weights_init = tf.truncated_normal_initializer(stddev=stddev)
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                      weights_initializer=weights_init,
                      activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
                with slim.arg_scope([slim.separable_conv2d],
                                weights_regularizer=depthwise_regularizer) as sc:
                    return sc

def ssd_feat_shapes_from_net(predictions, default_shapes=None):
    """Try to obtain the feature shapes from the prediction layers. The latter
    can be either a Tensor or Numpy ndarray.

    Return:
      list of feature shapes. Default values if predictions shape not fully
      determined.
    """
    feat_shapes = []
    for l in predictions:
        # Get the shape, from either a np array or a tensor.
        if isinstance(l, np.ndarray):
            shape = l.shape
        else:
            shape = l.get_shape().as_list()
        shape = shape[1:4]
        # Problem: undetermined shape...
        if None in shape:
            return default_shapes
        else:
            feat_shapes.append(shape)
    return feat_shapes

# =========================================================================== #
# Functional definition of VGG-based SSD 300.
# =========================================================================== #
def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]

# =========================================================================== #
# SSD loss function.
# =========================================================================== #
def lane_net_losses(logits, gt_maps,
                   negative_ratio=3.,
                   alpha=1.,
                   label_smoothing=0.,
                   device='/cpu:0',
                   scope=None):
    with tf.name_scope(scope, 'lane_net_losses'):
        lshape = tfe.get_shape(logits[0], 5)
        num_classes = lshape[-1]
        batch_size = lshape[0]

        # logits = tf.layers.flatten(logits)
        # gt = tf.layers.flatten(gt_maps)
        print (gt_maps.get_shape().as_list())
        print (logits.get_shape().as_list())

        # Add cross-entropy loss.
        with tf.name_scope('spase_softmax_cross_entropy'):
            loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=gt_maps,
                logits=logits
            )
            # loss = tf.nn.l2_loss(logits - gt_maps) 
            # loss = tf.nn.weighted_cross_entropy_with_logits(
                # targets=gt_maps,
                # logits=logits,
                # pos_weight=12
            # )
            loss = tf.reduce_mean(loss)
            # loss = tf.div(loss, batch_size, name='value')
            tf.losses.add_loss(loss)

