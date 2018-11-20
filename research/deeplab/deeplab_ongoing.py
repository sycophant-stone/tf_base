# -*- coding: UTF-8 -*-
import os
import sys
import math
import tensorflow as tf
#import common
import collections
import six
from tensorflow.contrib.slim.nets import resnet_utils
from deployment import model_deploy
import tensorflow.contrib.slim as slim
# for parsing debug
from deeplab.utils import input_generator

slim = tf.contrib.slim
prefetch_queue = slim.prefetch_queue
tfexample_decoder = slim.tfexample_decoder
dataset = slim.dataset
dataset_data_provider = slim.dataset_data_provider
dataset_data_provider = slim.dataset_data_provider
LOGITS_SCOPE_NAME = 'logits'
MERGED_LOGITS_SCOPE = 'merged_logits'
IMAGE_POOLING_SCOPE = 'image_pooling'
ASPP_SCOPE = 'aspp'
CONCAT_PROJECTION_SCOPE = 'concat_projection'
DECODER_SCOPE = 'decoder'





local_min_resize_value= None

local_max_resize_value= None

local_resize_factor= None
local_logits_kernel_size= 1

local_model_variant= "xception_65"

local_image_pyramid= None

local_add_image_level_feature= True

local_image_pooling_crop_size= None

local_aspp_with_batch_norm= True

local_aspp_with_separable_conv= True

local_multi_grid= None

local_depth_multiplier= 1.0

local_decoder_output_stride=4

local_decoder_use_separable_conv=True

local_merge_method='max'



OUTPUT_TYPE = 'semantic'
LABELS_CLASS = 'labels_class'
IMAGE = 'image'
HEIGHT = 'height'
WIDTH = 'width'
IMAGE_NAME = 'image_name'
LABEL = 'label'
ORIGINAL_IMAGE = 'original_image'
TEST_SET = 'test'


local_outputs_to_num_classes={'semantic': 21}
local_crop_size=[513, 513]
local_atrous_rates=[6, 12, 18]
local_output_stride=8


# from utils import input_generator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
'''
logger.info('This is a log info')
logger.debug('Debugging')
logger.warning('Warning exists')
logger.info('Finish')
'''
local_min_resize_value = None
local_max_resize_value = None
local_resize_factor = None
dataset_split = "train"
is_training = True
local_image_pyramid = None
# For `xception_65`# use atrous_rates = [12# 24# 36] if output_stride = 8# or
# rates = [6# 12# 18] if output_stride = 16. For `mobilenet_v2`# use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
atrous_rates = [6, 12, 18]

'''"jkcloud", "win10", "shiyan_ai" '''
GLB_ENV = "win10"

if GLB_ENV == "win10":
    print("WELCOM to Win10 env!!!")
    dataset_dir = "D:\\work\\stuff\\modules\\misc\\sprd_camera\\alg\\july\\tf_base\\research\\deeplab\\datasets\\pascal_voc_seg\\tfrecord"
    train_logdir = "D:\\work\\stuff\\modules\\misc\\sprd_camera\\alg\\july\\tf_base\\research\\deeplab\\datasets\\pascal_voc_seg\\output_new"
    tf_initial_checkpoint = "D:\\work\\stuff\\modules\\misc\\sprd_camera\\alg\\july\\tf_base\\research\\deeplab\\datasets\\pascal_voc_seg\\init_models\\deeplabv3_pascal_train_aug\\model.ckpt"
    eval_logdir="D:\\work\\stuff\\modules\\misc\\sprd_camera\\alg\\july\\tf_base\\research\\deeplab\\datasets\\pascal_voc_seg\\eval_output"
    # tf_initial_checkpoint = None
    checkpoint_dir = "D:\\work\\stuff\\modules\\misc\\sprd_camera\\alg\\july\\tf_base\\research\\deeplab\\datasets\\pascal_voc_seg\\output"
    eval_logdir = "D:\\work\\stuff\\modules\\misc\\sprd_camera\\alg\\july\\tf_base\\research\\deeplab\\datasets\\pascal_voc_seg\\eval_output"
elif GLB_ENV == "jkcloud":
    print("WELCOM to jkcloud env!!!")
    dataset_dir = "/work/tf_base/research/deeplab/datasets/pascal_voc_seg/tfrecord/"
    # Settings for logging.
    train_logdir = "/output/train_out"  # jikecloudåªæœ‰/outputå¯ä»¥ç”¨tensorboard
    # tf_initial_checkpoint = None
    tf_initial_checkpoint = "/work/tf_base/research/deeplab/datasets/pascal_voc_seg/init_models/deeplabv3_pascal_train_aug/model.ckpt"
    
    eval_logdir="/output/eval_out"
    # tf_initial_checkpoint = None
    checkpoint_dir = train_logdir

elif GLB_ENV == "shiyan_ai":
    print("WELCOM to shiyan.ai env!!!")
    dataset_dir = "/home/deeplearning/work/tf_base/research/deeplab/datasets/pascal_voc_seg/tfrecord/"
    # dataset_dir = "datasets/pascal_voc_seg/tfrecord/"
    train_logdir = "deeplab/output"  # shiyan.aiæ²¡æœ‰æ ¹ç›®å½•æƒé™
    # tf_initial_checkpoint=None
    tf_initial_checkpoint = "/home/deeplearning/work/tf_base/research/deeplab/init_models/deeplabv3_pascal_train_aug/model.ckpt"
else:
    raise ValueError('Please chose one Env to start!')


# -----------------------xception65ç½‘ç»œ----------------------

def fixed_padding(inputs, kernel_size, rate=1):
    """æ·»åŠ pad
    """
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)  # å’Œrateæœ‰å…³.
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg  # å–å‡ºkernel åŠå¾„,ç„¶åŽè®¡ç®—å‡ºèµ·å§‹ç»“æŸ.
    print("kernel_size_effective:%d, kernel_size:%d, rate:%d" % (kernel_size_effective, kernel_size, rate))
    print("pad_total:%d, pad_beg:%d, pad_end:%d" % (pad_total, pad_beg, pad_end))
    # åˆ©ç”¨tf.padåšpadding. pad_valueæ˜¯0.
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
    return padded_inputs


@slim.add_arg_scope
def separable_conv2d_same(inputs,
                          num_outputs,
                          kernel_size,
                          local_depth_multiplier,
                          stride,
                          rate=1,
                          use_explicit_padding=True,
                          regularize_depthwise=False,
                          scope=None,
                          **kwargs):
    """2Dçš„SAMEç±»åž‹çš„.
    stride>1ä¸”use_explicit_paddingæ˜¯True.å…ˆåšä¸€ä¸ªzero padding,ç„¶åŽæŽ¥ä¸€ä¸ªVALID padding.
    åŠŸèƒ½ç±»ä¼¼:
       net = slim.separable_conv2d(inputs, num_outputs, 3,
       local_depth_multiplier=1, stride=1, padding='SAME')
       net = resnet_utils.subsample(net, factor=stride)
    ä½†è¿™ä¸ªä¼šæœ‰evenæ—¶å€™çš„é”™è¯¯.

    """

    def _separable_conv2d(padding):
        """Wrapper for separable conv2d."""
        return slim.separable_conv2d(inputs,
                                     num_outputs,
                                     kernel_size,
                                     depth_multiplier=local_depth_multiplier,
                                     stride=stride,
                                     rate=rate,
                                     padding=padding,
                                     scope=scope,
                                     **kwargs)

    def _split_separable_conv2d(padding):
        """Splits separable conv2d into depthwise and pointwise conv2d."""
        outputs = slim.separable_conv2d(inputs,
                                        None,  # è¿™é‡Œåªåšdepthwise
                                        kernel_size,
                                        depth_multiplier=local_depth_multiplier,
                                        stride=stride,
                                        rate=rate,
                                        padding=padding,
                                        scope=scope + '_depthwise',
                                        **kwargs)
        return slim.conv2d(outputs,
                           num_outputs,
                           1,
                           scope=scope + '_pointwise',
                           **kwargs)

    if stride == 1 or not use_explicit_padding:
        if regularize_depthwise:
            outputs = _separable_conv2d(padding='SAME')
        else:
            outputs = _split_separable_conv2d(padding='SAME')
    else:
        '''stride!=1 å¹¶ä¸” use_explicit_padding'''
        inputs = fixed_padding(inputs, kernel_size, rate)
        if regularize_depthwise:
            outputs = _separable_conv2d(padding='VALID')
        else:
            outputs = _split_separable_conv2d(padding='VALID')

    return outputs


@slim.add_arg_scope
def xception_module(inputs,
                    depth_list,
                    skip_connection_type,
                    stride,
                    unit_rate_list=None,
                    rate=1,
                    activation_fn_in_separable_conv=False,
                    regularize_depthwise=False,
                    outputs_collections=None,
                    scope=None):
    """ xceptionæ¨¡å—åŒ…æ‹¬:
        'residual'å’Œ'shortcut'
        residualå«æœ‰separable conv 3x3
        shortcutå«æœ‰1x1 conv or not
        xceptionä¸ä½¿ç”¨max pooling.è€Œæ˜¯é‡‡ç”¨separable conv with striding.åŽŸå› æ˜¯å½“å‰max poolingä¸æ”¯æŒå¸¦æ´žæ“ä½œ.
        skip_connection_type: shortcutå’Œresidualçš„concatæ–¹æ³•,conv,sum,none, convæ˜¯shortcuté€šè·¯ç»è¿‡1x1å°å·ç§¯å’ŒresidualåŠ å’Œ,
                              sumæ˜¯residualå’ŒshortcutåŠ å’Œ.
                              noneåªé‡‡ç”¨residual.
    """
    if len(depth_list) != 3:
        raise ValueError('Expect three elements in depth_list.')
    if unit_rate_list:
        if len(unit_rate_list) != 3:
            raise ValueError('Expect three elements in unit_rate_list.')
    with tf.variable_scope(scope, 'xception_module', [inputs]) as sc:
        residual = inputs

        # åŠŸèƒ½å‡½æ•°,å¤„ç†reluåœ¨sperable convå‰è¿˜æ˜¯åŽ.
        def _separable_conv(features, depth, kernel_size, local_depth_multiplier,
                            regularize_depthwise, rate, stride, scope):
            if activation_fn_in_separable_conv:
                activation_fn = tf.nn.relu
            else:
                activation_fn = None
                features = tf.nn.relu(features)
            return separable_conv2d_same(features,
                                         depth,
                                         kernel_size,
                                         local_depth_multiplier=local_depth_multiplier,
                                         stride=stride,
                                         rate=rate,
                                         activation_fn=activation_fn,
                                         regularize_depthwise=regularize_depthwise,
                                         scope=scope)

        for i in range(3):
            residual = _separable_conv(residual, depth_list[i],
                                       kernel_size=3,
                                       local_depth_multiplier=1,
                                       regularize_depthwise=regularize_depthwise,
                                       rate=rate * unit_rate_list[i],
                                       stride=stride if i == 2 else 1,
                                       scope='separable_conv' + str(i + 1))
            print("[xception_module]:i:%d,input_stride:%d,stirde:%d,residual:%s" % (
            i, stride, stride if i == 2 else 1, residual))

        if skip_connection_type == 'conv':
            shortcut = slim.conv2d(inputs,
                                   depth_list[-1],
                                   [1, 1],
                                   stride=stride,
                                   activation_fn=None,
                                   scope='shortcut')
            print("xception_module[residual]", residual)
            print("xception_module[shortcut]", shortcut)
            outputs = residual + shortcut
        elif skip_connection_type == 'sum':
            outputs = residual + inputs
        elif skip_connection_type == 'none':
            outputs = residual
        else:  # None, è¡¨ç¤ºæ²¡æœ‰shortcutè¿™ä¸ªæ·å¾„
            raise ValueError('Unsupported skip connection type.')

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.name,
                                                outputs)


@slim.add_arg_scope
def stack_blocks_dense(net,
                       blocks,
                       output_stride=None,
                       outputs_collections=None):
    """Extract features for entry_flow, middle_flow, and exit_flow.
    args:
        net: tensor bhwc
        blocks: æè¿°xceptionçš„block
        output_stride: è¾“å‡ºçš„strides.
    """
    current_stride = 1  # åˆå§‹å€¼
    rate = 1
    for block in blocks:  # éåŽ†æ‰€æœ‰çš„blocks
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for idx, unit in enumerate(block.args):
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError('The output_stride can not be reached')
                with tf.variable_scope('unit_%d' % (idx + 1), values=[net]):
                    # å¤šæ¬¡æ‰§è¡Œxception_moduleæ¥æž„å»º,ç»“æŸæ¡ä»¶:
                    #    å¦‚æžœè¾¾åˆ°æœŸæœ›çš„output_strides,å°±å¯ä»¥ç»“æŸ,åŒæ—¶æ›´æ–°rate.
                    #    å¦‚æžœæ²¡æœ‰åˆ°output_strides,å°±å¤šæ‰§è¡Œå‡ ä¸ªxception_module
                    # netæ˜¯ä¸€ä¸ªtensor, bhwc
                    print("[stack_blocks_dense]:output_stride:%d,current_stride:%d" % (output_stride, current_stride))
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))  # xception_module
                        rate *= unit.get('stride', 1)
                    else:
                        net = block.unit_fn(net, rate=1, **unit)  # xception_module
                        current_stride *= unit.get('stride', 1)
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')

    return net


@slim.add_arg_scope
def xception(inputs,
             blocks,
             num_classes=None,
             is_training=True,
             global_pool=True,
             keep_prob=0.5,
             output_stride=None,
             reuse=None,
             scope=None):
    """æŠŠBlockæè¿°çš„ç½‘ç»œç»„ç»‡èµ·æ¥
    args:
        blocks: æè¿°äº†ä¸€ç³»åˆ—çš„xception ç½‘ç»œblock.
    """
    with tf.variable_scope(scope, 'xception',
                           [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + 'end_points'
        with slim.arg_scope([slim.conv2d,
                             slim.separable_conv2d,
                             xception_module,
                             stack_blocks_dense],
                            outputs_collections=end_points_collection):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                net = inputs
                if output_stride is not None:
                    if output_stride % 2 != 0:
                        raise ValueError('output_stride should be a mulitple of 2')
                    output_stride /= 2  # ç¼©è¿›
                # xceptionçš„entry flowå‰é¢è¿˜æœ‰ä¸¤ä¸ªconv
                net = resnet_utils.conv2d_same(net, 32, 3, stride=2, scope='entry_flow/conv1_1')
                net = resnet_utils.conv2d_same(net, 64, 3, stride=1, scope='entry_flow/conv1_2')

                # æŠ½å–blocksæè¿°çš„ç½‘è·¯
                net = stack_blocks_dense(net, blocks, output_stride)
                end_points = slim.utils.convert_collection_to_dict(end_points_collection, clear_collection=True)

                if global_pool:
                    net = tf.reduce_mean(net, [1, 2], name='global_pool', keepdim=True)
                    end_points['global_pool'] = net
                if num_classes:
                    net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training, scope='prelogits_dropout')
                    net = slim.conv2d(inputs=net, num_classes=num_classes, kernel_size=[1, 1], activation_fn=None,
                                      normalizer_fn=None, scope='logits')
                    end_points[sc.name + '/logits'] = net
                    end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """xceptionæ¨¡å—
    unit_fn: xceptionæ¨¡å—
    """


def xception_block(scope,
                   depth_list,
                   skip_connection_type,
                   activation_fn_in_separable_conv,
                   regularize_depthwise,
                   num_units,
                   stride,
                   unit_rate_list=None):
    """æž„å»ºxceptionå—.
    num_units: æè¿°ç›¸åŒçš„è¯¥å—æœ‰å¤šå°‘ä¸ª.

    """
    # BLOCK1_BUG: å¯åŠ¨å¼‚å¸¸.ä½¿ç”¨unit_rate_listæ—¶å€™,ä¼šæŠ¥'NoneType' object is not subscriptable
    # è¿™éƒ¨åˆ†è¦ç»™ä¸€ä¸ªé»˜è®¤å€¼,ä»¥é¿å…unit_rate_listæœªèµ‹å€¼å¯¼è‡´çš„å¼‚å¸¸.
    # unit_rate_list æ˜¯å¯¹åº”xceptionæ¨¡å—éœ€è¦çš„å‚æ•°.å†³å®šè¯¥æ¨¡å—çš„unit rate,åœ¨çš„exit flowä¸­ä¼šæœ‰å¯¹åº”çš„unit rate.è¿™é‡Œ
    # æ·»åŠ ä¸€ä¸ªé»˜è®¤å€¼.é¿å…NoneTypeå¯¼è‡´çš„å¼‚å¸¸.
    if unit_rate_list == None:
        unit_rate_list = [1, 1, 1]  # _DEFAULT_local_multi_grid
    print("scope:%s,stride:%d" % (scope, stride))
    return Block(scope, xception_module, [{
        'depth_list': depth_list,
        'skip_connection_type': skip_connection_type,
        'activation_fn_in_separable_conv': activation_fn_in_separable_conv,
        'regularize_depthwise': regularize_depthwise,
        'stride': stride,
        'unit_rate_list': unit_rate_list,
    }] * num_units)


def xception_65(inputs,
                num_classes=None,
                is_training=True,
                global_pool=True,
                keep_prob=0.5,
                output_stride=None,
                regularize_depthwise=False,
                local_multi_grid=None,
                reuse=None,
                scope='xception_65'):
    """æ­å»ºXception-65 æ¨¡åž‹
    """
    blocks = [
        xception_block('entry_flow/block1',
                       depth_list=[128, 128, 128],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('entry_flow/block2',
                       depth_list=[256, 256, 256],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('entry_flow/block3',
                       depth_list=[728, 728, 728],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),

        xception_block('middle_flow/block1',
                       depth_list=[728, 728, 728],
                       skip_connection_type='sum',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=16,
                       stride=1),

        xception_block('exit_flow/block1',
                       depth_list=[728, 1024, 1024],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('exit_flow/block2',
                       depth_list=[1536, 1536, 2048],
                       skip_connection_type='none',
                       activation_fn_in_separable_conv=True,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=1,
                       unit_rate_list=local_multi_grid),

    ]
    return xception(inputs,
                    blocks=blocks,
                    num_classes=num_classes,
                    is_training=is_training,
                    global_pool=global_pool,
                    keep_prob=keep_prob,
                    output_stride=output_stride,
                    reuse=reuse,
                    scope=scope)


# ---------------------------------------------

def cal_scaled_dim_val(dim, scale_coeff):
    """åˆ©ç”¨scale_coeffå¯¹dimç»´åšæˆscaleçš„ç»´åº¦.
       è¿™é‡Œåªæ˜¯ä¸€ä¸ªè®¡ç®—,è®¡ç®—scaleä¹‹åŽçš„ç»´åº¦æ•°.å…¶å®žå¹¶æ²¡æœ‰å®žé™…scaleæ•°æ®Tensorå¯¹è±¡.
    """
    if isinstance(dim, tf.Tensor):
        return tf.cast((tf.tofloat(dim) - 1.0) * scale_coeff + 1.0,
                       tf.int32)  # å…¶å®žè¿™é‡Œè¾¹çš„è¿™ä¸ª+1.0æ˜¯ä¸ºäº†å‘ä¸Šå–æ•´
    else:
        # BLOCK1_BUG: TypeError: Expected int32, got 33.0 of type 'float' instead.
        # å®šä½åˆ°æ˜¯è®¡ç®—scaled dimæ—¶å€™æ²¡æœ‰è½¬æˆintç±»åž‹.
        return int((float(dim) - 1.0) * scale_coeff + 1.0)


def xception_arg_scope(weight_decay=0.00004,
                       batch_norm_decay=0.9997,
                       batch_norm_epsilon=0.001,
                       batch_norm_scale=True,
                       weights_initializer_stddev=0.09,
                       activation_fn=tf.nn.relu,
                       regularize_depthwise=False,
                       use_batch_norm=True):
    """ç”Ÿæˆxception65 ä½¿ç”¨çš„arg_scope.

    """
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
    }
    if regularize_depthwise:
        depthwise_regularizer = slim.l2_regularizer(weight_decay)
    else:
        depthwise_regularizer = None
    with slim.arg_scope(
            [slim.conv2d, slim.separable_conv2d],
            weights_initializer=tf.truncated_normal_initializer(stddev=weights_initializer_stddev),
            activation_fn=activation_fn,
            normalizer_fn=slim.batch_norm if use_batch_norm else None):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d],
                                weights_regularizer=slim.l2_regularizer(weight_decay)):
                with slim.arg_scope([slim.separable_conv2d], weights_regularizer=depthwise_regularizer) as arg_sc:
                    return arg_sc


def _preprocess_zero_mean_unit_range(inputs):
    """æŠŠå›¾åƒæ•°æ®è½¬æˆ-1åˆ°1
    """
    return tf.to_float(inputs) / 255 * 2.0 - 1.0


def get_network(network_name, preprocess_images, arg_scope=None):
    """get networkçš„å‡½æ•°ä»¥åŠå¯¹åº”çš„å‚æ•°
    """
    arg_scope = arg_scope or xception_arg_scope()  # åœ¨xceptioné»˜è®¤å‚æ•°é…ç½®åŸºç¡€ä¸Š

    # if preprocess_images==True:
    # preprocess_function =_preprocess_zero_mean_unit_range
    def network_fn(inputs, *args, **kwargs):
        with slim.arg_scope(arg_scope):
            return xception_65(_preprocess_zero_mean_unit_range(inputs),
                               *args, **kwargs)

    return network_fn


# BLOCK1_BUG: å¯¹extract_featuresè¿”å›žçš„featuresç»asppåŽåšconcatæ—¶é‡åˆ°tensor sizeä¸ä¸€è‡´é—®é¢˜.
#            å®šä½æ ¹æºæ˜¯: `stack_blocks_dense`å‡½æ•°ä¼šä»Ž1æˆæ¯”ä¾‹è·¨åˆ°åˆ°æœŸæœ›çš„output_stride,å¦‚æžœæ»¡è¶³çš„è¯,ä»¥åŽçš„blockå°±ä¼šç”¨
#                         stride=1çš„å‚æ•°åŽ»åšäº†.è¿™ä¹Ÿæ˜¯åˆç†çš„,å¦åˆ™çš„è¯ç½‘ç»œçš„strideä¼šå¤§äºŽæœŸæœ›çš„output_stride.
#                         å¦‚æžœä¸æ˜Žç™½çš„è¯,è¦æ³¨æ„å®ƒä»¬è¿™äº›entry,middle,exit flowæ˜¯ çº§è”çš„.

def local_extract_features(
        features,
        output_stride=8,
        local_multi_grid=None,
        local_depth_multiplier=1.0,
        weight_decay=0.0001,
        reuse=None,
        is_training=False,
        fine_tune_batch_norm=False,
        regularize_depthwise=False):
    """å¯¹äºŽç‰¹å®šçš„æ¨¡åž‹æŠ½å–features

    """
    # åšxceptionçš„local_model_variant.
    arg_scope = xception_arg_scope(weight_decay=weight_decay, batch_norm_decay=0.9997,
                                   batch_norm_epsilon=1e-3,
                                   batch_norm_scale=True,
                                   regularize_depthwise=regularize_depthwise)
    temp_network = get_network(network_name="exception", preprocess_images=True, arg_scope=arg_scope)
    features, endpoints = temp_network(inputs=features,
                                       num_classes=None,
                                       is_training=is_training,
                                       global_pool=False,
                                       output_stride=output_stride,
                                       ### BLOCK1_BUG: æ­¤å¤„ä¸èƒ½æ˜¯8,åº”è¯¥é‡‡ç”¨ä¼ å…¥çš„å€¼16.
                                       local_multi_grid=None,
                                       reuse=reuse,
                                       scope='xception_65')
    return features, endpoints

local_image_pooling_crop_size=None
def extract_features(features,
                     model_options=None,
                     weight_decay=0.0001,
                     reuse=None,
                     is_training=False,
                     fine_tune_batch_norm=False):
    """æå–ç‰¹å¾å›¾å’Œend_point.
       è¿”å›žå€¼:
       1. concat_logits, å®ƒæ˜¯ä¸€ç³»åˆ—çš„èžåˆ.
       é¦–å…ˆ,ä¸»å¹²è¾“å‡ºçš„feature maps.ç»è¿‡image pooläº§ç”Ÿä¸€ä¸ªç»“æžœ; ç»è¿‡1x1çš„conväº§ç”Ÿä¸€ä¸ªç»“æžœ; ç»è¿‡rates(6,12,18)çš„asppäº§ç”Ÿä¸€ç»„ç»“æžœ.
       ç„¶åŽ,è¿™äº›ç»“æžœä¼šconcatæˆä¸€ä¸ªè¾“å‡º.
       end_points
    """
    # æå–
    features, end_points = local_extract_features(
        features,
        output_stride=local_output_stride,
        local_multi_grid=local_multi_grid,
        local_depth_multiplier=local_depth_multiplier,
        weight_decay=weight_decay,
        reuse=reuse,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm)

    if not local_aspp_with_batch_norm:
        # asppä¸éœ€è¦batch norm,ç›´æŽ¥è¿”å›žfeatures
        # æˆ‘ä»¬çŸ¥é“batch normæ˜¯åœ¨æ¿€æ´»å‡½æ•°ä¹‹å‰,å¯¹featuresåšçš„.è®©å…¶å½’ä¸€åŒ–åˆ°0~1ä¹‹é—´.
        return features, end_points
    else:
        batch_norm_params = {
            'is_training': is_training and fine_tune_batch_norm,
            'decay': 0.9997,
            'epsilon': 1e-5,
            'scale': True,
        }
    # slim.arg_scopeå¯¹ç»™å®šçš„opå­˜å‚¨å…¶param
    # æž„å»ºfigure5é‡Œè¾¹çš„ Block4ä¹‹åŽå¤„ç†çš„ASPPéƒ¨åˆ†.
    with slim.arg_scope(
            [slim.conv2d, slim.separable_conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            padding='SAME',
            stride=1,
            reuse=reuse):
        # ç›®çš„æ˜¯è¦åšbatch norm
        with slim.arg_scope(
                [slim.batch_norm],
                **batch_norm_params):
            depth = 256
            branch_logits = []
            # step 1 åšä¸€ä¸ªpooling
            # èžåˆBéƒ¨åˆ†(image pooling).
            if local_add_image_level_feature:
                if local_crop_size is not None:
                    local_image_pooling_crop_size = local_crop_size
                    # è®¡ç®—poolingçš„scale
                    pooling_height = cal_scaled_dim_val(local_image_pooling_crop_size[0],
                                                        1.0 / local_output_stride)
                    pooling_width = cal_scaled_dim_val(local_image_pooling_crop_size[1],
                                                       1.0 / local_output_stride)
                    # åŠ poolingå±‚
                    image_feature_pooled = slim.avg_pool2d(features,
                                                           [pooling_height, pooling_width],
                                                           # è¿™ä¸ªåº”è¯¥æ˜¯kernelçš„size
                                                           [1, 1],  # è¿™ä¸ªåº”è¯¥æ˜¯hwçš„strides
                                                           padding='VALID')
                    # crop size ä¹Ÿéœ€è¦åšä¸€ä¸‹scale
                    resize_height = cal_scaled_dim_val(local_crop_size[0],
                                                       1.0 / local_output_stride)
                    resize_width = cal_scaled_dim_val(local_crop_size[1],
                                                      1.0 / local_output_stride)
                else:
                    # æ²¡æœ‰cropçš„size,åšä¸€ä¸ªglobalçš„pooling
                    pooling_height = tf.shape(features)[0]
                    pooling_width = tf.shape(features)[1]
                    image_feature_pooled = tf.reduce_mean(
                        features,
                        axis=[1, 2])[:, tf.newaxis,
                                           tf.newaxis]  # åœ¨featuresåŸºç¡€ä¸Šå†æ·»åŠ ä¸¤ä¸ªç»´åº¦,ä½†æ˜¯è¿™ä¸¤ä¸ªç»´åº¦è¿˜æ²¡æœ‰å…¶ä»–çš„å¡«å……å€¼.
                    resize_height = pooling_height
                    resize_width = pooling_width
                # æ·»åŠ ä¸€ä¸ª1x1çš„å·ç§¯
                image_feature = slim.conv2d(
                    image_feature_pooled, depth, 1, scope=IMAGE_POOLING_SCOPE)
                # æ’å€¼æˆresizeçš„feature map
                image_feature = tf.image.resize_bilinear(image_feature, size=[resize_height, resize_width],
                                                         align_corners=True)

                if isinstance(resize_height, tf.Tensor):
                    resize_height = None
                if isinstance(resize_width, tf.Tensor):
                    resize_width = None

                image_feature.set_shape([None, resize_height, resize_width, depth])
                print("image_feature:%s " % (image_feature))
                branch_logits.append(image_feature)

            # step 2 å¯¹featuresåš1x1å·ç§¯,æ³¨æ„æ­¤å¤„å¹¶ä¸æ˜¯å¯¹ç»è¿‡poolingçš„image_featureåš1x1å·ç§¯.
            # èžåˆAéƒ¨åˆ†(ASPP) éœ€è¦1x1
            print("features:", features)
            temp = slim.conv2d(features, depth, 1, scope=ASPP_SCOPE + str(0))
            branch_logits.append(temp)

            print("[extract_features]:local_atrous_rates:", local_atrous_rates)
            # ASPP,çš„é‡‘å­—å¡”æ¯å±‚é‡‡ç”¨ä¸åŒçš„atrous rates,æ­¤å¤„æž„å»ºè¿™ç»„atrous pyramid
            # èžåˆAéƒ¨åˆ†(ASPP) éœ€è¦3x3 å¸¦artous.
            if local_atrous_rates:
                # 3x3å·ç§¯
                for i, rate in enumerate(local_atrous_rates, 1):
                    scope = ASPP_SCOPE + str(i)
                    # å¦‚æžœé‡‡ç”¨å¯åˆ†ç¦»å·ç§¯
                    if local_aspp_with_separable_conv:
                        aspp_features = split_separable_conv2d(
                            features,
                            filters=depth,
                            rate=rate,
                            weight_decay=weight_decay,
                            scope=scope)
                    else:
                        aspp_features = slim.conv2d(features, depth, 3, rate=rate, scope=scope)

                    branch_logits.append(aspp_features)

            for itm in branch_logits:
                print("[extract_features]:branch_logits:", itm)
            # æŠŠè¿™äº›ç»„ä»¶ç»„åˆèµ·æ¥
            concat_logits = tf.concat(branch_logits, 3)  # åœ¨é€šé“ä¸Šå¢žåŠ äº†.å¢žåŠ äº†é€šé“
            print("[extract_features]:concat the branch_logits: ", concat_logits)
            concat_logits = slim.conv2d(
                concat_logits, depth, 1, scope=CONCAT_PROJECTION_SCOPE)
            print("[extract_features]:after slim.conv2d: ", concat_logits)
            concat_logits = slim.dropout(concat_logits, keep_prob=0.9, is_training=is_training,
                                         scope=CONCAT_PROJECTION_SCOPE + '_dropout')
            print("[extract_features]:after slim.dropout: ", concat_logits)

    return concat_logits, end_points


# -------------------------------------------------------------------------------

def split_separable_conv2d(inputs,
                           filters,
                           kernel_size=3,
                           rate=1,
                           weight_decay=0.00004,
                           depthwise_weights_initializer_stddev=0.33,
                           pointwise_weights_initializer_stddev=0.06,
                           scope=None):
    """æŠŠä¸€ä¸ªseparable covn2dè½¬åŒ–æˆ depthwise å’Œ pointwiseçš„conv2d
        depthwise_filterã€‚ä¸€ä¸ªå¼ é‡ï¼Œæ•°æ®ç»´åº¦æ˜¯å››ç»´[filter_height,filter_width,in_channels,channel_multiplier]ï¼Œå¦‚1ä¸­æ‰€è¿°ï¼Œä½†æ˜¯å·ç§¯æ·±åº¦æ˜¯1
        pointwise_filter ä¸€ä¸ªå¼ é‡, ç»´åº¦æ˜¯[1, 1, in_ch*ch_muli, out_ch]
       æ³¨æ„:
            è¯¥å‡½æ•°å’Œtf.layers.separable_conv2dæ˜¯æœ‰åŒºåˆ«çš„. è¯¥å‡½æ•°ä¼šåœ¨depthwiseå’Œpointwiseé—´åŠ ä¸Šä¸€ä¸ªæ¿€æ´»å‡½æ•°

        filters:
            æ˜¯è¾“å‡ºçš„ä¸ªæ•°,å¯ç†è§£ä¸ºnum_outputs
    """

    """æ³¨æ„slim.separable_conv2dçš„outputså‚æ•°
            å¦‚æžœoutputså‚æ•°æ˜¯None,slim.separable_conv2dä¼šè·³è¿‡point_wiseé˜¶æ®µ
            (num_outputs: The number of pointwise convolution output filters. If is
             None, then we skip the pointwise convolution stage.)
    """
    outputs = slim.separable_conv2d(
        inputs,
        None,  # åªåšdepth_wiseé˜¶æ®µ,è·³è¿‡point_wiseé˜¶æ®µ
        kernel_size=kernel_size,
        depth_multiplier=1,  # DMæ˜¯1
        rate=rate,
        weights_initializer=tf.truncated_normal_initializer(
            stddev=depthwise_weights_initializer_stddev
        ),
        weights_regularizer=None,
        scope=scope + '_depthwise')
    return slim.conv2d(
        outputs,  # ä¸Šä¸€å±‚çš„è¾“å‡º,æŽ¥ç€åš1x1xfiltersçš„point wiseé˜¶æ®µ
        filters,  # è¾“å‡ºçš„filterçš„ä¸ªæ•°
        1,
        weights_initializer=tf.truncated_normal_initializer(
            stddev=pointwise_weights_initializer_stddev
        ),
        weights_regularizer=slim.l2_regularizer(weight_decay),
        scope=scope + '_pointwise')

# A map from feature extractor name to the network name scope used in the
# ImageNet pretrained versions of these models.
name_scope = {
    'mobilenet_v2': 'MobilenetV2',
    'resnet_v1_50': 'resnet_v1_50',
    'resnet_v1_50_beta': 'resnet_v1_50',
    'resnet_v1_101': 'resnet_v1_101',
    'resnet_v1_101_beta': 'resnet_v1_101',
    'xception_41': 'xception_41',
    'xception_65': 'xception_65',
    'xception_71': 'xception_71',
}
# A dictionary from network name to a map of end point features.
DECODER_END_POINTS = 'decoder_end_points'
networks_to_feature_maps = {
    'mobilenet_v2': {
        DECODER_END_POINTS: ['layer_4/depthwise_output'],
    },
    'resnet_v1_50': {
        DECODER_END_POINTS: ['block1/unit_2/bottleneck_v1/conv3'],
    },
    'resnet_v1_50_beta': {
        DECODER_END_POINTS: ['block1/unit_2/bottleneck_v1/conv3'],
    },
    'resnet_v1_101': {
        DECODER_END_POINTS: ['block1/unit_2/bottleneck_v1/conv3'],
    },
    'resnet_v1_101_beta': {
        DECODER_END_POINTS: ['block1/unit_2/bottleneck_v1/conv3'],
    },
    'xception_41': {
        DECODER_END_POINTS: [
            'entry_flow/block2/unit_1/xception_module/'
            'separable_conv2_pointwise',
        ],
    },
    'xception_65': {
        DECODER_END_POINTS: [
            'entry_flow/block2/unit_1/xception_module/'
            'separable_conv2_pointwise',
        ],
    },
    'xception_71': {
        DECODER_END_POINTS: [
            'entry_flow/block2/unit_1/xception_module/'
            'separable_conv2_pointwise',
        ],
    },
}

def refine_by_decoder(features,
                      end_points,
                      decoder_height,
                      decoder_width,
                      local_decoder_use_separable_conv=False,
                      local_model_variant=None,
                      weight_decay=0.0001,
                      reuse=None,
                      is_training=False,
                      fine_tune_batch_norm=False):
    """ æ·»åŠ decoderéƒ¨åˆ†

    """
    batch_norm_params = {
        'is_training': is_training and fine_tune_batch_norm,
        'decay': 0.9997,
        'epsilon': 1e-5,
        'scale': True,
    }

    with slim.arg_scope(
            [slim.conv2d, slim.separable_conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            padding='SAME',
            stride=1,
            reuse=reuse):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with tf.variable_scope(DECODER_SCOPE, DECODER_SCOPE, [features]):
                feature_list = networks_to_feature_maps[
                    local_model_variant][DECODER_END_POINTS]
                if feature_list is None:
                    tf.logging.info('[refine_by_decoder]:Not found')
                    return features
                else:
                    decoder_features = features
                    for i, name in enumerate(feature_list):
                        decoder_features_list = [decoder_features]
                        if 'mobilenet' in local_model_variant:
                            feature_name = name
                        else:
                            feature_name = '{}/{}'.format(
                                name_scope[local_model_variant],
                                name)
                        decoder_features_list.append(
                            slim.conv2d(
                                end_points[feature_name],
                                48,
                                1,
                                scope='feature_projection' + str(i)))

                        # resize
                        for j, feature in enumerate(decoder_features_list):
                            decoder_features_list[j] = tf.image.resize_bilinear(
                                feature, [decoder_height, decoder_width],
                                align_corners=True)
                            h = (None if isinstance(decoder_height, tf.Tensor)
                                 else decoder_height)
                            w = (None if isinstance(decoder_width, tf.Tensor)
                                 else decoder_width)
                            decoder_features_list[j].set_shape([None, h, w, None])
                        decoder_depth = 256
                        print("[refine_by_decoder]: decoder_height:%s, decoder_width:%s,decoder_features_list:%s" %(decoder_height, decoder_width, decoder_features_list))
                        if local_decoder_use_separable_conv:
                            decoder_features = split_separable_conv2d(
                                tf.concat(decoder_features_list, 3),
                                filters=decoder_depth,
                                rate=1,
                                weight_decay=weight_decay,
                                scope='decoder_conv0')
                            decoder_features = split_separable_conv2d(
                                decoder_features,
                                filters=decoder_depth,
                                rate=1,
                                weight_decay=weight_decay,
                                scope='decoder_conv1')
                        else:
                            num_convs = 2
                            decoder_features = slim.repeat(
                                tf.concat(decoder_features_list, 3),
                                num_convs,
                                slim.conv2d,
                                decoder_depth,
                                3,
                                scope='decoder_conv' + str(i))

                return decoder_features


# -------------------------------------------------------------------------------

def get_branch_logits(features,
                      num_class,
                      atrous_rates=None,
                      local_aspp_with_batch_norm=False,
                      kernel_size=1,
                      weight_decay=0.0001,
                      reuse=None,
                      scope_suffix=''):
    """ ä»Žæ¨¡åž‹ä¸­èŽ·å¾—logits
        xception åŽæŽ¥asppçš„è¾“å‡ºæ˜¯logits.
    """
    # å½“asppåº”ç”¨bnæ—¶,åœ¨extract_featuresä¹‹å‰å°±ç”¨ä¸Šaspp,è¿™é‡Œé‡‡ç”¨1x1çš„conv
    if local_aspp_with_batch_norm or atrous_rates is None:
        if kernel_size != 1:
            # å¦‚æžœæœ‰bnçš„atrousä¹Ÿæœ‰,è¯æ˜Žæ˜¯aspp.éœ€è¦åŠ ä¸€ä¸ª1x1çš„conv
            raise ValueError('kernel size must be 1')
        atrous_rates = [1]

    with slim.arg_scope(
            # slim.arg_scopeä½œç”¨å°±æ˜¯æˆ‘ä»¬å¯ä»¥é¢„å…ˆå†™ä¸€äº›ä¸ªå‚æ•°,ä»¥åŽå†è°ƒç”¨opçš„æ—¶å€™å¯ä»¥ä¸ç”¨å†™äº†,å‡å°‘ä¹¦å†™.
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
            reuse=reuse):
        with tf.variable_scope(LOGITS_SCOPE_NAME, LOGITS_SCOPE_NAME, [features]):
            branch_logits = []  # å¯¹æ¯ä¸ªatrousç®—ä¸€ä¸ªåˆ†æ”¯,å¯¹äºŽæ¯ä¸ªåˆ†æ”¯éƒ½å­˜ä¸€ä¸ªlogits.
            for i, rate in enumerate(atrous_rates):
                scope = scope_suffix
                if i:
                    scope += '_%d' % i

                branch_logits.append(
                    slim.conv2d(
                        features,
                        num_class,
                        kernel_size=kernel_size,
                        rate=rate,
                        activation_fn=None,
                        normalizer_fn=None,
                        scope=scope))

    return tf.add_n(branch_logits)


def _get_logits(images, weight_decay=0.0001, reuse=None, is_training=False,
                fine_tune_batch_norm=False):
    """ç”Ÿæˆlogitsç½‘ç»œ.è¯¥ç½‘ç»œåº”ç”¨åˆ°aspp,atrous spatial pyramid pooling.
    """
    # æå–featureså’Œend_points.
    features, end_points = extract_features(
        images,
        weight_decay=weight_decay,
        reuse=reuse,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm)

    # å¦‚æžœdecoder æœ‰ç‰¹æ®Šå®šä¹‰çš„stride.éœ€è¦å¯¹decoder sizeåšscale
    if local_decoder_output_stride is not None:
        if local_crop_size is None:
            height = tf.shape(images)[1]
            width = tf.shape(images)[2]
        else:
            # cropå­˜åœ¨
            height, width = local_crop_size

        # æ±‚decoderä½¿ç”¨çš„size.è¿™ä¸ªæ˜¯ç»è¿‡local_decoder_output_strideä¹‹åŽçš„.
        decoder_height = cal_scaled_dim_val(height, 1.0 / local_decoder_output_stride)
        decoder_width = cal_scaled_dim_val(width, 1.0 / local_decoder_output_stride)

        # å¯¹featuresåšé‡æ–°refine
        # æ·»åŠ decodeéƒ¨åˆ†,ä¹‹å‰çš„deeplabé‡‡ç”¨äº†crf,åœ¨deeplabV3ä¸­æ²¡æœ‰ä½¿ç”¨crf.åˆ©ç”¨sep conv2dæ›¿ä»£.
        # åªä¸è¿‡è¿™é‡Œä¼šä½¿ç”¨ä¹‹å‰xceptionçš„ä¸­é—´äº§ç‰©åšèžåˆ.
        features = refine_by_decoder(
            features,
            end_points,
            decoder_height=decoder_height,
            decoder_width=decoder_width,
            local_decoder_use_separable_conv=local_decoder_use_separable_conv,  # ä½¿ç”¨ç¦»æ•£å·ç§¯
            local_model_variant=local_model_variant,
            weight_decay=weight_decay,
            reuse=reuse,
            is_training=is_training,
            fine_tune_batch_norm=fine_tune_batch_norm)

    # èŽ·å¾—batchçš„logits
    # batch_normalizationä½œç”¨:
    # åœ¨æ¿€æ´»å‡½æ•°ä¹‹å‰çš„bnæ¨¡å—,å®ƒæŽ¥å—wx+bè®¡ç®—çš„featureä½œä¸ºè¾“å…¥.å¯ä»¥åšåˆ°å¦‚ä¸‹å‡ ç‚¹:
    # 1. æé«˜æ¢¯åº¦ä¼ æ’­æ•°åº¦,å°†æ‰€æœ‰è¾“å‡ºå½’ä¸€åŒ–åˆ°0~1.é¿å…æ¢¯åº¦æ¶ˆå¤±.
    # 2. æé«˜æ¨¡åž‹çš„æ”¶æ•›é€Ÿåº¦.(å½’ä¸€åŒ–åˆ°0~1,æ‰€æœ‰çš„featureéƒ½æ˜¯)
    # 3. å‡å°‘æ¨¡åž‹å¯¹å‚æ•°åˆå§‹åŒ–çš„å½±å“.(å½’ä¸€åŒ–åˆ°0~1)
    outputs_to_logits = {}
    for output in sorted(local_outputs_to_num_classes):
        outputs_to_logits[output] = get_branch_logits(
            features,
            local_outputs_to_num_classes[output],
            local_atrous_rates,
            local_aspp_with_batch_norm=local_aspp_with_batch_norm,  # batch normalization
            weight_decay=weight_decay,
            reuse=reuse,
            scope_suffix=output)  # scope_suffixåŽç¼€

    return outputs_to_logits


def multi_scale_logits(images,
                       local_image_pyramid,
                       weight_decay=0.0001,
                       is_training=False,
                       fine_tune_batch_norm=False):
    """

    :param images: 输入图
    :param local_image_pyramid:图像金字塔
    :param weight_decay: r2的权重是否衰减
    :param is_training: train or test
    :param fine_tune_batch_norm: 是否是fine tune模式
    :return:返回网络的预测,图像经过xception+aspp+decoder+upsampling4
    """
    if not local_image_pyramid:
        local_image_pyramid = [1.0]  # list
    # crop size
    crop_height = (
        local_crop_size[0]
        if local_crop_size else tf.shape(images)[1])
    crop_width = (
        local_crop_size[1]
        if local_crop_size else tf.shape(images)[0])
    # local_decoder_output_stride æ˜¯åœ¨decoderå•å…ƒ,æç‚¼åˆ†å‰²ç»“æžœæ—¶å€™ä½¿ç”¨çš„ input/outputçš„æ¯”
    logits_output_stride = (
        local_decoder_output_stride or local_output_stride)

    logit_height = cal_scaled_dim_val(crop_height, max(1.0, max(
        local_image_pyramid)) / logits_output_stride)  # è¿™ä¸ªlocal_image_pyramidå…¶å®žä¸æ˜¯å¾ˆæ¸…æ¥šå®ƒçš„å«ä¹‰,æ˜¯åŒå°ºå¯¸çš„imagesç»„è¿˜æ˜¯downsizeåŽimages
    logit_width = cal_scaled_dim_val(crop_width, max(1.0, max(
        local_image_pyramid)) / logits_output_stride)  # 2018-09-17: è¿™ä¸ªlocal_image_pyramideæ˜¯åŒ…å«äº†ä¸€ç»„å›¾åƒç¼©æ”¾çš„fractor.å¹¶ä¸æ˜¯å›¾åƒæœ¬èº«.

    outputs_to_scales_to_logits = {
        k: {}
        for k in local_outputs_to_num_classes
    }

    # step 1  对于每一个缩略图
    for image_scale in local_image_pyramid:
        if image_scale != 1.0:
            # 不是原图,需要缩放
            # 有了缩放因子,需要计算对应的缩放尺寸
            scaled_height = cal_scaled_dim_val(crop_height, image_scale)
            scaled_width = cal_scaled_dim_val(crop_width, image_scale)
            scaled_crop_size = [scaled_height, scaled_width]
            # 有了缩放尺寸,需要对原图做缩放了
            scaled_images = tf.image.resize_bilinear(images, scaled_crop_size, align_corners=True)

            if local_crops_size:
                scaled_images.set_shape([None, scaled_height, scale_width,
                                         3])  # 如果需要crop size的话,我们把scaled_images reshape成3个chn的.
        else:
            # 原图
            scaled_crop_size = local_crop_size
            scaled_images = images

        # 用做过scale的尺寸替换参数中的crop_size,然后生成网络
        #updated_options = local_replace(crop_size=scaled_crop_size)
        loca_crop_size = scaled_crop_size
        outputs_to_logits = _get_logits(
            scaled_images,
            weight_decay=weight_decay,
            reuse=tf.AUTO_REUSE,
            is_training=is_training,
            fine_tune_batch_norm=fine_tune_batch_norm)
        # 此时拿到结果.对结果做一个reshape,以便和其他的scale pyramid做融合使尺寸是合理的.
        for output in sorted(outputs_to_logits):
            outputs_to_logits[output] = tf.image.resize_bilinear(
                outputs_to_logits[output],
                [logit_height, logit_width],
                align_corners=True)

        # 只有一层pyramid,就可以返回
        if len(local_image_pyramid) == 1:
            for output in sorted(local_outputs_to_num_classes):
                # 第k个scaler fractor对应的LOGITS_SCOPE_NAME,AKA,"logits"
                outputs_to_scales_to_logits[output][MERGED_LOGITS_SCOPE] = outputs_to_logits[output]

            return outputs_to_scales_to_logits

        # 如果有多个pyramid fractor,需要按照对应的标签保存
        for output in sorted(local_outputs_to_num_classes):
            outputs_to_scales_to_logits[output]['logits_%.2f' % image_scale] = outputs_to_logits[output]

    # 把多个pyramid fractor融合
    # 需要新创建一个维度,该维度为了融合使用
    for output in local_outputs_to_num_classes:
        all_logits = [
            tf.expand_dims(logits, axis=4)
            for logits in outputs_to_scales_to_logits[output].values()
        ]
        # 在这个新维度上做concat( 理解为连接)
        all_logits = tf.concat(all_logits, axis=4)
        # 根据不同的融合方法采用不同的tf的融合方法
        merge_fn = (
            tf.reduce_max
            if local_merge_method == 'max' else tf.reduce_mean)
        # 在新增维度上融合.
        outputs_to_scales_to_logits[output][MERGED_LOGITS_SCOPE] = merge_fn(all_logits, axis=4)

    return outputs_to_scales_to_logits


# ------------------------------------------------------------------------------------------
# train


num_clones = 1  # 'Number of clones to deploy.')

clone_on_cpu = False  # 'Use CPUs to deploy clones.')

num_replicas = 1  # 'Number of worker replicas.')

startup_delay_steps = 15  # 'Number of training steps between replicas startup.')

num_ps_tasks = 0  # 'The number of parameter servers. If the value is 0# then '                     'the parameters are handled locally by the worker.')

master = ''  # '='BNS name of the tensorflow server')

task = 0  # 'The task ID.')

log_steps = 10  # Display logging information at every log_steps.')

save_interval_secs = 1200  # 'How often# in seconds# we save the model to disk.')

save_summaries_secs = 60  # 'How often# in seconds# we compute the summaries.')

save_summaries_images = False  # 'Save sample inputs# labels# and semantic predictions as '                     'images to summary.')

# Settings for training strategy.

learning_policy = 'poly'  # 'Learning rate policy for training.')

# Use 0.007 when training on PASCAL augmented training set# train_aug. When
# fine-tuning on PASCAL trainval set# use learning rate=0.0001.
base_learning_rate = 0.0001  # 'The base learning rate for model training.')

learning_rate_decay_factor = 0.1  # 'The rate to decay the base learning rate.')

learning_rate_decay_step = 2000  # 'Decay the base learning rate at a fixed step.')

learning_power = 0.9  # 'The power value used in the poly learning policy.')

training_number_of_steps = 30000  # 'The number of steps used for training')

momentum = 0.9  # 'The momentum value to use')

# When fine_tune_batch_norm=True# use at least batch size larger than 12
# (batch size more than 16 is better). Otherwise# one could use smaller batch
# size and set fine_tune_batch_norm=False.
train_batch_size = 8  # 'The number of images in each batch during training.')

# For weight_decay# use 0.00004 for MobileNet-V2 or Xcpetion model variants.
# Use 0.0001 for ResNet model variants.
weight_decay = 0.00004  # 'The value of the weight decay for training.')

train_crop_size = [513, 513]  # 'Image crop size [height# width] during training.')

last_layer_gradient_multiplier = 1.0  # 'The gradient multiplier for last layers# which is used to '                   'boost the gradient of last layers if the value > 1.')

upsample_logits = True  # 'Upsample logits during training.')
# Settings for fine-tuning the network.


# Set to False if one does not want to re-use the trained classifier weights.
initialize_last_layer = True  # 'Initialize the last layer.')

last_layers_contain_logits_only = False  # 'Only consider logits as last layers or not.')

slow_start_step = 0  # 'Training model with small learning rate for few steps.')

slow_start_learning_rate = 1e-4  # 'Learning rate employed during slow start.')

# Set to True if one wants to fine-tune the batch norm parameters in DeepLabv3.
# Set to False and use small batch size to save GPU memory.
fine_tune_batch_norm = True  # 'Fine tune the batch norm parameters or not.')

min_scale_factor = 0.5  # 'Mininum scale factor for data augmentation.')

max_scale_factor = 2.  # 'Maximum scale factor for data augmentation.')

scale_factor_step_size = 0.25  # 'Scale factor step size for data augmentation.')

output_stride = 16  # 'The ratio of input to output spatial resolution.')

# Dataset settings.
dataset_name = 'pascal_voc_seg'  # 'Name of the segmentation dataset.')

train_split = 'train'  # 'Which split of the dataset to be used for training')

# --train utils
_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'labels_class': ('A semantic segmentation label whose size matches image.'
                     'Its values range from 0 (background) to num_classes.'),
}
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    [
        'splits_to_sizes',
        'num_classes',  # åˆ†ç±»,åŒ…å«èƒŒæ™¯ç±».ä¾‹å¦‚pascalæ˜¯20åˆ†ç±»+1ä¸ªèƒŒæ™¯
        'ignore_label'
    ]
)
##BLOCK1_BUG:
#    æç¤ºinput_queue.dequeue()å‡ºé”™è¯¯. å®šä½åˆ°è¿™é‡Œ,åº”è¯¥æ˜¯datasetæœ¬èº«è§£æžå°±æœ‰é”™è¯¯.é‡‡ç”¨é”™è¯¯çš„æ•°æ®æ ¼å¼è§£æžå¾—åˆ°é”™è¯¯çš„å€¼.
#    å› æ­¤ä¼šå¯¼è‡´get samplesæ—¶å€™æŠ¥é”™è¯¯.
# _PASCAL_VOC=datasetDescriptor(
#    splits_to_size={
#        'train':2975,
#        'val':500,
#    },
#    num_classes=19,
#    ignore_label=255,
# )

_PASCAL_VOC = DatasetDescriptor(
    splits_to_sizes={
        'train': 1464,
        'train_aug': 10582,
        'trainval': 2913,
        'val': 1449,
    },
    num_classes=21,
    ignore_label=255,
)

# ---------
# å‚æ•°è‡ªå®šä¹‰

train_batch_size = 1


#####--



def get_dataset(dataset_name, split_name, dataset_dir):
    """èŽ·å¾—slim datasetå®žä¾‹
    """
    splite_size = _PASCAL_VOC.splits_to_sizes
    num_classes = _PASCAL_VOC.num_classes
    ignore_label = _PASCAL_VOC.ignore_label

    if split_name not in splite_size:
        raise ValueError('data split name %s not recognized' % split_name)
    # file pattern
    file_pattern = os.path.join(dataset_dir, '%s-*' % split_name)

    # TF è§£ç åè®®
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/filename': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature(
            (), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature(
            (), tf.int64, default_value=0),
        'image/width': tf.FixedLenFeature(
            (), tf.int64, default_value=0),
        'image/segmentation/class/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/segmentation/class/format': tf.FixedLenFeature(
            (), tf.string, default_value='png'),
    }
    items_to_handlers = {
        'image': tfexample_decoder.Image(
            image_key='image/encoded',
            format_key='image/format',
            channels=3),
        'image_name': tfexample_decoder.Tensor('image/filename'),
        'height': tfexample_decoder.Tensor('image/height'),
        'width': tfexample_decoder.Tensor('image/width'),
        'labels_class': tfexample_decoder.Image(
            image_key='image/segmentation/class/encoded',
            format_key='image/segmentation/class/format',
            channels=1),
    }

    print("[get_dataset]:splite_size:%s,num_classes:%s,ignore_label:%s" % (splite_size, num_classes, ignore_label))
    print("[get_dataset]:file_pattern:%s" % (file_pattern))
    print("[get_dataset]:keys_to_features:%s" % (keys_to_features))
    print("[get_dataset]:items_to_handlers:%s" % (items_to_handlers))
    decoder = tfexample_decoder.TFExampleDecoder(
        keys_to_features,
        items_to_handlers)
    print("[dataset.Dataset]:data_sources:", file_pattern)
    print("[dataset.Dataset]:decoder:", decoder)
    print("[dataset.Dataset]:num_samples:", splite_size[split_name])
    print("[dataset.Dataset]:items_to_descriptions:", _ITEMS_TO_DESCRIPTIONS)
    print("[dataset.Dataset]:ignore_label:", ignore_label)
    print("[dataset.Dataset]:num_classes:", num_classes)
    print("[dataset.Dataset]:name:", dataset_name)
    print("[dataset.Dataset]:multi_label:", True)
    return dataset.Dataset(
        data_sources=file_pattern,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=splite_size[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        ignore_label=ignore_label,
        num_classes=num_classes,
        name=dataset_name,
        multi_label=True)


# ---
# input get samples



def _get_data(data_provider, dataset_split):
    """data_providerçš„list_items()ä¸­å«æœ‰æ•°æ®å†…å®¹.
    """
    if "labels_class" not in data_provider.list_items():
        raise ValueError("labels_class not in dataset")
    print("[_get_data]:data_provider.list_items():", data_provider.list_items())
    image, height, width = data_provider.get(["image", "height", "width"])
    logger.debug('_get_data,image:%s' % image)
    logger.debug('_get_data,height:%s' % height)
    logger.debug('_get_data,widht:%s' % width)

    if "image_name" in data_provider.list_items():
        image_name, = data_provider.get(["image_name"])  # è¦æœ‰","å·.
    else:
        image_name = tf.constant('')
    label = None
    if dataset_split != "test":
        label, = data_provider.get(["labels_class"])
    return image, label, image_name, height, width


_PROB_OF_FLIP = 0.5


# ---preprocesså‡½æ•°ç»„
def resolve_shape(tensor, rank=None, scope=None):
    """è¿”å›žè¯¥tensorçš„full shape
    """
    with tf.name_scope(scope, 'resolve_shape', [tensor]):
        if rank is not None:
            shape = tensor.get_shape().with_rank(rank).as_list()
        else:
            shape = tensor.get_shape().as_list()
        if None in shape:
            shape_dynamic = tf.shape(tensor)
            for i in range(len(shape)):
                if shape[i] is None:
                    shape[i] = shape_dynamic[i]

        return shape


def resize_to_range_1(image,
                      label=None,
                      min_size=None,
                      max_size=None,
                      factor=None,
                      align_corners=True,
                      label_layout_is_chw=False,
                      scope=None,
                      method=tf.image.ResizeMethod.BILINEAR):
    """

    """
    print("[resize_to_range]: label:", label)
    print("[resize_to_range]: min_size:", min_size)
    print("[resize_to_range]: max_size:", max_size)
    print("[resize_to_range]: factor:", factor)
    print("[resize_to_range]: label_layout_is_chw:", label_layout_is_chw)
    print("[resize_to_range]: scope:", scope)
    print("[resize_to_range]: image:", image)
    print("[resize_to_range]: new_size:", new_size)
    print("[resize_to_range]: method:", method)
    print("[resize_to_range]: align_corners:", align_corners)
    with tf.name_scope(scope, 'resize_to_range', [image]):
        new_tensor_list = []
        min_size = tf.to_float(min_size)
        if max_size is not None:
            max_size = tf.to_float(max_size)
            if factor is not None:
                max_size = (max_size + (factor - (max_size - 1) % factor) % factor - factor)

        [orig_height, orig_width, _] = resolve_shape(image, rank=3)
        orig_height = tf.to_float(orig_height)
        orig_width = tf.to_float(orig_width)
        orig_min = tf.minimum(orig_height, orig_width)

        # è®©originçš„æœ€å°å°ºå¯¸æ»¡è¶³é™å®šçš„æœ€å°å°ºå¯¸,è¿™é‡Œæ‰©å¤§ä¸€ä¸ªå€æ•°.
        large_scale_factor = min_size / orig_min
        large_height = tf.to_int32(tf.ceil(orig_height * large_scale_factor))
        large_width = tf.to_int32(tf.ceil(orig_width * large_scale_factor))
        large_size = tf.stack([large_height, large_width])

        new_size = large_size
        if max_size is not None:
            orig_max_size = tf.maximum(orig_height, orig_width)
            small_scale_factor = max_size / orig_max_size
            small_height = tf.to_int32(tf.ceil(orig_height * small_scale_factor))
            small_width = tf.to_int32(tf.ceil(orig_width * small_scale_factor))
            small_size = tf.stack([small_height, small_width])
            new_size = tf.cond(
                tf.to_float(tf.reduce_max(large_size)) > max_size,
                # é™åˆ¶æ»¡è¶³æœ€å°sizeçš„æƒ…å†µä¸‹,ä¸è¦è¶…è¿‡æœ€å¤§size.
                lambda: small_size,
                lambda: large_size)
        if factor is not None:
            new_size += (factor - (new_size - 1) % factor) % factor  # factorçš„å€æ•°.
        new_tensor_list.append(tf.image.resize_images(
            image, new_size, method=method, align_corners=align_corners))

        if label is not None:
            if label_layout_is_chw:
                # Input label has shape [channel, height, width].
                resized_label = tf.expand_dims(label, 3)
                resized_label = tf.image.resize_nearest_neighbor(
                    resized_label, new_size, align_corners=align_corners)
                resized_label = tf.squeeze(resized_label, 3)
            else:
                # Input label has shape [height, width, channel].
                resized_label = tf.image.resize_images(
                    label, new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                    align_corners=align_corners)
                new_tensor_list.append(resized_label)
        else:
            new_tensor_list.append(None)
        return new_tensor_list


def get_random_scale(min_scale_factor, max_scale_factor, step_size):
    """èŽ·å–éšæœºscaleå€¼.ç”¨äºŽscaleå›¾åƒ.
    """
    if min_scale_factor == max_scale_factor:
        return tf.to_float(min_scale_factor)
    # ç¬¬ä¸€æ¬¡æ—¶å€™,éšæœº
    if step_size == 0:
        return tf.random_uniform(shape=[1], minval=min_scale_factor, maxval=max_scale_factor)
    # å…¶ä»–é‡‡ç”¨æ­¥é•¿ç›¸å…³çš„ç¦»æ•£å€¼.
    else:
        num_steps = int(
            (max_scale_factor - min_scale_factor) / step_size + 1)  # step_sizeä¸ªå¤§å°æ—¶å€™,å¯¹åº”æœ‰num_stepsä¸ª
        scale_factors = tf.lin_space(min_scale_factor, max_scale_factor, num_steps)
        shuffled_scale_factors = tf.random_shuffle(scale_factors)
        return shuffled_scale_factors[0]


def randomly_scale_image_and_label(image, label, scale=1.0):
    """éšæœºç¼©æ”¾å›¾åƒ
    """
    if scale == 1.0:
        return image, label
    img_shape = tf.shape(image)
    new_dims = tf.to_int32(tf.to_float([img_shape[0], img_shape[1]]) * scale)
    image = tf.squeeze(
        tf.image.resize_bilinear(tf.expand_dims(image, 0),
                                 new_dims,
                                 align_corners=True),
        [0])
    if label is not None:
        label = tf.squeeze(tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_dims, align_corners=True),
                           [0])
    return image, label


def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width, pad_value):
    """åœ¨imageå®½é«˜ä¸ŠåŠ pad,å…¶å€¼æ˜¯pad_value
    args:
        offset_width, æ˜¯padingåœ¨å·¦ä¾§çš„å®½åº¦.
        offset_height,æ˜¯padingåœ¨ä¸Šé¢çš„é«˜åº¦.
        target_width,æ˜¯æ€»çš„å®½åº¦.
        target_height,æ˜¯æ€»é«˜åº¦.
    """
    image -= pad_value  # ä»¥åŽè¿˜ä¼šåŠ ä¸Šæ¥.å› ä¸ºpadé»˜è®¤æ˜¯0,ä»¥åŽå¯¹æ‰€æœ‰å›¾ç‰‡åƒç´ åŠ ä¸€ä¸ªpad_value
    img_shape = tf.shape(image)
    img_height = img_shape[0]
    img_width = img_shape[1]
    right_width = target_width - img_width - offset_width
    bottom_height = target_height - img_height - offset_height

    height_params = tf.stack([offset_height, bottom_height])
    width_params = tf.stack([offset_width, right_width])
    chn_params = tf.stack([0, 0])

    paddings = tf.stack([height_params, width_params, chn_params])
    padded = tf.pad(image, paddings)
    padded += pad_value
    return padded


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    """å¯¹hwåšcrop
    """
    original_shape = tf.shape(image)
    if len(image.get_shape().as_list()) != 3:
        raise ValueError("should be hwc")
    original_chns = image.get_shape().as_list()[2]
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])  # ä¿ç•™chns,chnsä¸åšcrop
    offset = tf.stack([offset_height, offset_width, 0])  # chnsä¸æ¶‰åŠoffset
    image = tf.slice(image, offset, cropped_shape)
    image = tf.reshape(image, cropped_shape)  # çŠ¯é”™ç‚¹
    image.set_shape([crop_height, crop_width, original_chns])
    return image


def random_crop(image_list, crop_height, crop_width):
    """éšæœºè£å‰ª
    args:
        image_list, ä¼ å…¥ä¸€ç»„list.å¯ä»¥æ˜¯[preimage,label]
    """
    img_shape = tf.shape(image_list[0])
    img_h = img_shape[0]
    img_w = img_shape[1]
    max_offset_h = tf.reshape(img_h - crop_height + 1, [])
    max_offset_w = tf.reshape(img_w - crop_width + 1, [])
    offset_height = tf.random_uniform([], maxval=max_offset_h, dtype=tf.int32)
    offset_width = tf.random_uniform([], maxval=max_offset_w, dtype=tf.int32)
    return [_crop(image, offset_height, offset_width, crop_height, crop_width) for image in image_list]


def flip_dim(tensor_list, prob=0.5, dim=1):
    """éšæœºflip
    """
    random_val = tf.random_uniform([])

    def flip():
        flipped = []
        for tensor in tensor_list:
            flipped.append(tf.reverse_v2(tensor, [dim]))
        return flipped

    is_flipped = tf.less_equal(random_val, prob)  # è®¾ç½®ä¸€ä¸ªéšæœºé˜ˆå€¼,å…¶å®žå’Œdropoutçš„éšæœºé˜ˆå€¼ç±»ä¼¼.
    outputs = tf.cond(is_flipped, flip, lambda: tensor_list)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    outputs.append(is_flipped)

    return outputs  # é‡Œè¾¹ä¼šæœ‰ä¸¤ä¸ªå†…å®¹, imageç­‰ä»¥åŠæ˜¯å¦æ˜¯flipped.


##

def preprocess_image_and_label(image,
                               label,
                               crop_height,
                               crop_width,
                               local_min_resize_value=None,
                               local_max_resize_value=None,
                               local_resize_factor=None,
                               min_scale_factor=1.,
                               max_scale_factor=1.,
                               scale_factor_step_size=0,
                               ignore_label=255,
                               is_training=True,
                               local_model_variant=None):
    """è¿”å›ž: origin image
            preprocess image
            label( groud truth)
    """
    # ä¿å­˜origin image
    origin_image = image
    process_image = tf.cast(image, tf.float32)
    print("[preprocess_image_and_label] local_min_resize_value:", local_min_resize_value)
    print("[preprocess_image_and_label] local_max_resize_value:", local_max_resize_value)
    if label is not None:
        label = tf.cast(label, tf.int32)
    if local_min_resize_value is not None or local_max_resize_value is not None:
        print("[preprocess_image_and_label]: before resize_to_range")
        [process_image, label] = resize_to_range_1(image=process_image, label=label, min_size=local_min_resize_value,
                                                   max_size=local_max_resize_value,
                                                   factor=local_resize_factor, align_corners=True)
        original_image = tf.identity(process_image)  # originæè¿°çš„å˜æˆäº†resizeä¹‹åŽçš„.
        print("[preprocess_image_and_label]: after resize_to_range")

    # éšæœºç¼©æ”¾ ä»¥ è¾¾åˆ°æ•°æ®å¢žå¼º

    if is_training:
        argu_scale = get_random_scale(min_scale_factor, max_scale_factor, scale_factor_step_size)
        process_image, label = randomly_scale_image_and_label(process_image, label, argu_scale)
        process_image.set_shape([None, None, 3])  # 3ä¸ªchn
    print("[preprocess_image_and_label]:randomly_scale_image_and_label'process_image:", process_image)
    image_shape = tf.shape(process_image)
    image_height = image_shape[0]
    image_width = image_shape[1]

    target_height = image_height + tf.maximum(crop_height - image_height, 0)
    target_width = image_width + tf.maximum(crop_width - image_width, 0)
    meanpixel = [127.5, 127.5, 127.5]
    meanpixel = tf.reshape(meanpixel, [1, 1, 3])
    # pad
    processed_image = pad_to_bounding_box(process_image, 0, 0, target_height, target_width, meanpixel)
    if label is not None:
        label = pad_to_bounding_box(label, 0, 0, target_height, target_width, ignore_label)
    print("[preprocess_image_and_label]:pad_to_bounding_box'process_image:", process_image)
    # éšæœºè£å‰ª
    if is_training and label is not None:
        process_image, label = random_crop([process_image, label], crop_height, crop_width)
    print("[preprocess_image_and_label]:random_crop'process_image:", process_image)
    process_image.set_shape([crop_height, crop_width, 3])
    if label is not None:
        label.set_shape([crop_height, crop_width, 1])

    # éšæœºå·¦å³é¢ å€’
    if is_training:
        process_image, label, _ = flip_dim([process_image, label], _PROB_OF_FLIP,
                                           dim=1)  # è¿™ä¸ª`_`å·æè¿°çš„æ˜¯is_flippedä¿¡æ¯,è¿™é‡Œæˆ‘ä»¬ä¸å…³å¿ƒ,å°±çœç•¥æŽ‰äº†.
    print("[preprocess_image_and_label]:flip_dim'process_image:", process_image)
    return origin_image, process_image, label  # !!!!!!è¯·æ³¨æ„è¿™ä¸ªæ‹¼å†™é”™è¯¯:::processed_image, label


def get_samples(dataset,
                crop_size,
                batch_size,
                local_min_resize_value=None,
                local_max_resize_value=None,
                local_resize_factor=None,
                min_scale_factor=1.,
                max_scale_factor=1.,
                scale_factor_step_size=0,
                num_readers=1,
                num_threads=1,
                dataset_split=None,
                is_training=True):
    """
    1. è¿”å›žraw data
    2. é¢„å¤„ç† raw data
    3. batching é¢„å¤„ç†äº§ç”Ÿçš„data,å…¶ç»“æžœå¯ç›´æŽ¥è¢«ç”¨ä½œtrain,test
    """
    print("[get_samples]:dataset", dataset)
    print("[get_samples]:crop_size", crop_size)
    print("[get_samples]:batch_size", batch_size)
    print("[get_samples]:local_min_resize_value", local_min_resize_value)
    print("[get_samples]:local_max_resize_value", local_max_resize_value)
    print("[get_samples]:local_resize_factor", local_resize_factor)
    print("[get_samples]:min_scale_factor", min_scale_factor)
    print("[get_samples]:max_scale_factor", max_scale_factor)
    print("[get_samples]:scale_factor_step_size", scale_factor_step_size)
    print("[get_samples]:num_readers", num_readers)
    print("[get_samples]:num_threads", num_threads)
    print("[get_samples]:dataset_split", dataset_split)
    print("[get_samples]:is_training", is_training)

    data_provider = dataset_data_provider.DatasetDataProvider(dataset, num_readers,
                                                              num_epochs=None if is_training else 1,
                                                              shuffle=is_training)
    image, label, image_name, height, width = _get_data(data_provider, dataset_split)
    print("[get_samples]:image:", image)
    print("[get_samples]:label:", label)
    print("[get_samples]:image_name:", image_name)
    print("[get_samples]:height:", height)
    print("[get_samples]:width:", width)
    print("[get_samples]:label.shape.ndims:%d,label.shape.dims[2]:%d" % (label.shape.ndims, label.shape.dims[2]))
    if label is not None:
        if label.shape.ndims == 2:
            label = tf.expand_dims(label, 2)
        elif label.shape.ndims == 3 and label.shape.dims[2] == 1:
            pass
    else:
        raise ValueError('Input label shape must be [height, width], or '
                         '[height, width, 1].')
        label.set_shape([None, None, 1])  # æ³¨æ„ç¼©è¿›ç­‰çº§
    print("[get_samples]:before preprocess_image_and_label")
    original_image, image, label = preprocess_image_and_label(
        image,
        label,
        crop_height=crop_size[0],
        crop_width=crop_size[1],
        local_min_resize_value=local_min_resize_value,
        local_max_resize_value=local_max_resize_value,
        local_resize_factor=local_resize_factor,
        min_scale_factor=min_scale_factor,
        max_scale_factor=max_scale_factor,
        scale_factor_step_size=scale_factor_step_size,
        ignore_label=dataset.ignore_label,
        is_training=is_training)
    sample = {
        "image": image,
        "image_name": image_name,
        "height": height,
        "width": width
    }
    if label is not None:
        sample["label"] = label
    if not is_training:
        sample["original_image"] = original_image
    print("[get_samples]:after preprocess original_image:", original_image)
    print("[get_samples]:after preprocess image:", image)
    print("[get_samples]:after preprocess label:", label)
    print("[get_samples]:after preprocess sample:", sample)
    return tf.train.batch(
        sample,
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=32 * batch_size,
        allow_smaller_final_batch=not is_training,
        dynamic_pad=True)


# ---


def add_softmax_cross_entropy_loss_for_each_scale(scales_to_logits,
                                                  labels,
                                                  num_classes,
                                                  ignore_label,
                                                  loss_weight=1.0,
                                                  upsample_logits=True,
                                                  scope=None):
    """æž„å»ºloss function
    """
    if labels is None:
        print("[add_softmax_cross_entropy_loss_for_each_scale]: label is needed.")
        raise ValueError('No label for softmax cross entropy loss.')
    print("[add_softmax_cross_entropy_loss_for_each_scale]:scales_to_logits is a map: %s", scales_to_logits)
    for scale, logits in six.iteritems(scales_to_logits):
        print("[add_softmax_cross_entropy_loss_for_each_scale]: scale:%s, logits:%s" % (scale, logits))
        loss_scope = None
        if scope:
            loss_scope = '%s_%s' % (scope, scale)
            print("loss_scope:%s", loss_scope)
        if upsample_logits:
            print("Label is not downsampled, and instead we upsample logits.")
            print("label:%s,after resolve label:%s" % (labels, resolve_shape(labels, 4)))
            logits = tf.image.resize_bilinear(logits, resolve_shape(labels, 4)[1:3], align_corners=True)
            scaled_labels = labels
        else:
            print("Label is downsampled to the same size as logits.")
            print("logits:%s,after resolve logits:%s" % (logits, resolve_shape(logits, 4)))
            scaled_labels = tf.image.resize_nearest_neighbor(labels, resolve_shape(logits, 4)[1:3], align_corners=True)
        scaled_labels = tf.reshape(scaled_labels, shape=[-1])  # shape=[-1]å°†æŠŠç›®æ ‡å±•å¹³æˆ1Dçš„å°ºå¯¸.
        not_ignore_mask = tf.to_float(tf.not_equal(scaled_labels, ignore_label)) * loss_weight
        one_hot_labels = slim.one_hot_encoding(scaled_labels, num_classes, on_value=1.0, off_value=0.0)
        tf.losses.softmax_cross_entropy(
            one_hot_labels,
            tf.reshape(logits, shape=[-1, num_classes]),
            weights=not_ignore_mask,
            scope=loss_scope)


def get_model_learning_rate(
        learning_policy, base_learning_rate, learning_rate_decay_step,
        learning_rate_decay_factor, training_number_of_steps, learning_power,
        slow_start_step, slow_start_learning_rate):
    """èŽ·å¾—modelçš„learning rate.
    args:
        learning_policy: å­¦ä¹ ç­–ç•¥,åˆ†stepå’Œpolyä¸¤ç§.
        learning_rate_decay_step: ä»¥å›ºå®šå¤§å°decayå­¦ä¹ çŽ‡
        stepæ¨¡å¼:
        current_learning_rate = base_learning_rate *learning_rate_decay_factor ^ (global_step / learning_rate_decay_step)
        polyæ¨¡å¼:
        current_learning_rate = base_learning_rate * (1 - global_step / training_number_of_steps) ^ learning_power
        slow_start_step: `å¼€å§‹å‡ æ­¥ç”¨æ¯”è¾ƒå°çš„å­¦ä¹ çŽ‡`å¯¹åº”çš„å¼€å§‹çš„æ­¥æ•°
        slow_start_learning_rate: `å¼€å§‹å‡ æ­¥ç”¨è¾ƒå°çš„å­¦ä¹ çŽ‡`å¯¹åº”çš„å­¦ä¹ çŽ‡
    """
    global_step = tf.train.get_or_create_global_step()
    if learning_policy == 'step':
        learning_rate = tf.train.exponential_decay(base_learning_rate,
                                                   global_step,
                                                   learning_rate_decay_step,
                                                   learning_rate_decay_factor)
    elif learning_policy == 'poly':
        learning_rate = tf.train.polynomial_decay(base_learning_rate,
                                                  global_step,
                                                  training_number_of_steps,
                                                  end_learning_rate=0,
                                                  power=learning_power)
    else:
        raise ValueError('Unknown learning policy.')
    print(
        "[get_model_learning_rate]:global_step:%s,slow_start_step:%s,current_learning_rate:%s,slow_start_learing_rate:%s" % (
        global_step, slow_start_step, learning_rate, slow_start_learning_rate))
    return tf.where(global_step < slow_start_step, slow_start_learning_rate,
                    learning_rate)


def get_model_gradient_multipliers(last_layers, last_layer_gradient_multiplier):
    """ä¸åŒçš„layeræ‹¥æœ‰ä¸åŒæ¢¯åº¦ç³»æ•°.ä¾¿äºŽæ”¶æ•›.
    args:
        last_layers:æœ€åŽä¸€ä¸ªlayers.
    """
    gradient_multipliers = {}
    print("[get_model_gradient_multipliers]:all model variables")
    for var in slim.get_model_variables():
        print("[get_model_gradient_multipliers]:var:", var)
        if 'biases' in var.op.name:
            print("Double the learning rate for biases.")
            gradient_multipliers[var.op.name] = 2.
        print("Use larger learning rate for last layer variables.")
        for layer in last_layers:
            print("[get_model_gradient_multipliers]: last layers:", layer)
            if layer in var.op.name and 'biases' in var.op.name:
                print("last layer's biase")
                gradient_multipliers[var.op.name] = 2 * last_layer_gradient_multiplier
            elif layer in var.op.name:
                print("other last layers")
                gradient_multipliers[var.op.name] = last_layer_gradient_multiplier
                break

    return gradient_multipliers


def get_model_init_fn(train_logdir,
                      tf_initial_checkpoint,
                      initialize_last_layer,
                      last_layers,
                      ignore_missing_vars=False):
    """ä»Žcheckpointå¾—åˆ°æ¨¡åž‹çš„åˆå§‹åŒ–å˜é‡
    args:
        initialize_last_layer: æ˜¯å¦ç”¨checkpointå€¼åˆå§‹åŒ–æœ€åŽä¸€å±‚.è¿™é‡Œåœ¨fine-tuneæ—¶å€™ä¸éœ€è¦
        ignore_missing_vars: å¿½ç•¥checkpointä¸­çš„missingçš„å˜é‡
    """
    if tf_initial_checkpoint is None:
        tf.logging.info('Not initializing the model from a checkpoint.')
        print("[get_model_init_fn]:Not initializing the model from a checkpoint")
        return None
    if tf.train.latest_checkpoint(train_logdir):
        tf.logging.info('Ignoring initialization; other checkpoint exists')
        print("[get_model_init_fn]:Ignoring initialization; other checkpoint exists")
        return None
    tf.logging.info('Initializing model from path: %s', tf_initial_checkpoint)
    print("[get_model_init_fn]:train_logdir", train_logdir)
    print("[get_model_init_fn]:tf_initial_checkpoint", tf_initial_checkpoint)
    print("[get_model_init_fn]:initialize_last_layer", initialize_last_layer)
    print("[get_model_init_fn]:last_layers", last_layers)
    print("[get_model_init_fn]:ignore_missing_vars", ignore_missing_vars)
    exclude_list = ['global_step']
    print("[get_model_init_fn]:Variables that will not be restored.")
    if not initialize_last_layer:
        exclude_list.extend(last_layers)
    print("[get_model_init_fn]:Excluded var:", exclude_list)
    variables_to_restore = slim.get_variables_to_restore(exclude=exclude_list)
    if variables_to_restore:
        print("[get_model_init_fn]:variables_to_restore", variables_to_restore)
        return slim.assign_from_checkpoint_fn(
            tf_initial_checkpoint,
            variables_to_restore,
            ignore_missing_vars=ignore_missing_vars)

    return None


def _build_deeplab(inputs_queue, outputs_to_num_classes, ignore_labels):
    """æž„å»ºdeeplabç½‘ç»œ
    inputs_queue:
            è¾“å…¥sample
    outputs_num_classes:
            å½“å‰æ˜¯å‡ åˆ†ç±»çš„ç½‘ç»œ

    return:
            è¿”å›ždeeplabç½‘ç»œ
    """
    print("[_build_deeplab]:inputs_queue", inputs_queue)
    print("[_build_deeplab]:outputs_to_num_classes", outputs_to_num_classes)
    print("[_build_deeplab]:ignore_labels", ignore_labels)
    samples = inputs_queue.dequeue()
    print("[_build_deeplab]:inputs_queue:%s,samples:%s" % (inputs_queue, samples))
    print("[_build_deeplab]:fine_tune_batch_norm:%s" % (fine_tune_batch_norm))
    # æ·»åŠ ä¸€äº›åŠ©è®°åå­—
    samples["image"] = tf.identity(samples["image"], name="image")
    samples["label"] = tf.identity(samples["label"], name="label")

    # setup
    #model_options = common.ModelOptions(
    #    outputs_to_num_classes=outputs_to_num_classes,
    #    crop_size=train_crop_size,
    #    atrous_rates=atrous_rates,
    #    output_stride=output_stride)

    # å‡ çŽ‡è¡¨è¾¾å¼(å…¶å®žæ˜¯softmaxçš„è¾“å‡º,å¯è®¤ä¸ºæ˜¯æ¦‚çŽ‡)
    outputs_to_scales_to_logits = multi_scale_logits(
        samples["image"],
        local_image_pyramid=local_image_pyramid,  # local_image_pyramid=NULL
        weight_decay=weight_decay,  # 4e-05
        is_training=True,
        fine_tune_batch_norm=fine_tune_batch_norm)
    # æ·»åŠ ä¸€äº›åŠ©è®°åå­—
    output_type_dict = outputs_to_scales_to_logits["semantic"]
    output_type_dict["merged_logits"] = tf.identity(
        output_type_dict["merged_logits"],
        name="semantic")

    for output, num_classes in six.iteritems(outputs_to_num_classes):
        # softmax
        add_softmax_cross_entropy_loss_for_each_scale(
            outputs_to_scales_to_logits[output],
            samples["label"],
            num_classes,
            ignore_label=ignore_labels,
            loss_weight=1.0,
            upsample_logits=upsample_logits,  # Upsample logits during training
            scope=output)

    return outputs_to_scales_to_logits


def train():
    tf.logging.set_verbosity(tf.logging.INFO)
    config = model_deploy.DeploymentConfig(
        num_clones=num_clones,
        clone_on_cpu=clone_on_cpu,
        replica_id=task,
        num_replicas=num_replicas,
        num_ps_tasks=num_ps_tasks)

    clone_batch_size = train_batch_size // num_clones

    dataset = get_dataset(
        dataset_name,  # åˆ†å‰²datasetsçš„åå­—,æ˜¯pascal_voc_segè¿˜æ˜¯ä»€ä¹ˆ.
        train_split,  # å­—ç¬¦ä¸²,'train_split'æˆ–è€…'train'
        dataset_dir)  # datasetè·¯å¾„
    tf.gfile.MakeDirs(train_logdir)
    tf.logging.info('Training on %s set', train_split)

    with tf.Graph().as_default() as graph:
        with tf.device(config.inputs_device()):
            '''
            samples=get_samples(
                # ä»Žæ•°æ®é›†ä¸­æ‹¿åˆ°æ ·æœ¬
                dataset,
                train_crop_size,
                train_batch_size,
                local_min_resize_value=local_min_resize_value,
                local_max_resize_value=local_max_resize_value,
                local_resize_factor=local_resize_factor,
                min_scale_factor=min_scale_factor,
                max_scale_factor=max_scale_factor,
                scale_factor_step_size=scale_factor_step_size,
                dataset_split=train_split,
                is_training=True)
            '''
            samples = input_generator.get(
                dataset,
                [513, 513],
                clone_batch_size,
                min_resize_value=local_min_resize_value,
                max_resize_value=local_max_resize_value,
                resize_factor=local_resize_factor,
                min_scale_factor=min_scale_factor,
                max_scale_factor=max_scale_factor,
                scale_factor_step_size=scale_factor_step_size,
                dataset_split=train_split,
                is_training=True,
                model_variant="xception_65")
            # slim.prefetch_queueç”Ÿæˆä¸€ä¸ªqueueå®žä¾‹.
            print("[train]: get samples params:")
            print("[train]: dataset:", dataset)
            print("[train]: train_crop_size:", train_crop_size)
            print("[train]: train_batch_size:", train_batch_size)
            print("[train]: local_min_resize_value=local_min_resize_value:", local_min_resize_value)
            print("[train]: local_max_resize_value=local_max_resize_value:", local_max_resize_value)
            print("[train]: local_resize_factor=local_resize_factor:", local_resize_factor)
            print("[train]: min_scale_factor=min_scale_factor:", min_scale_factor)
            print("[train]: max_scale_factor=max_scale_factor:", max_scale_factor)
            print("[train]: scale_factor_step_size=scale_factor_step_size:", scale_factor_step_size)
            print("[train]: dataset_split:", train_split)
            print("[train]: is_training=", is_training)
            print("[train]: samples:", samples)

            inputs_queue = prefetch_queue.prefetch_queue(samples, capacity=128 * config.num_clones)
            # samples_try=inputs_queue.dequeue()
            # print("[train]:samples_try",samples_try)
            print("[train]: num_clones:%s,samples:%s,inputs_queue:%s" % (config.num_clones, samples, inputs_queue))

        with tf.device(config.variables_device()):
            global_step = tf.train.get_or_create_global_step()  # ä¸ºå½“å‰å›¾èŽ·å¾—(æœ‰å¿…è¦çš„è¯åŽ»åˆ›å»º)ä¸€ä¸ªå…¨å±€æ­¥æ•°è®¡æ•°çš„tensor,ä¸€ä¸ªgraphåªæœ‰ä¸€ä¸ªè¿™æ ·çš„tensor.
            print("[train]:Define the model and create clones.")
            model_fn = _build_deeplab
            model_args = (inputs_queue, {
                "semantic": dataset.num_classes
            }, dataset.ignore_label)
            print("[train]:model_fn", model_fn)
            print("[train]:model_args", model_args)
            clones = model_deploy.create_clones(config=config, model_fn=model_fn, args=model_args)

            # æ”¶é›†ç¬¬ä¸€ä¸ªcloneçš„updates,å¯èƒ½æœ‰bnå˜é‡çš„update.
            first_clone_scope = config.clone_scope(0)
            print("tf.Graphkeys.UPDATE_OPS", tf.GraphKeys.UPDATE_OPS)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        summaries=set(tf.get_collection(tf.GraphKeys.SUMMARIES,first_clone_scope))
        # 添加模型的变量
        #for model_var in slim.get_model_variables():
        #    summaries.add(tf.summary.histogram(model_var.op.name,model_var))

        if save_summaries_images:
            summary_image=graph.get_tensor_by_name( ('%s/%s:0' %(first_clone_scope,"image")).strip('/'))
            summaries.add(tf.summary.image('sample/%s'%("image"),summary_image))
            first_clone_label = graph.get_tensor_by_name(
            ('%s/%s:0' % (first_clone_scope, "label")).strip('/'))
            pixel_scaling = max(1, 255 // dataset.num_classes)
            summary_label = tf.cast(first_clone_label * pixel_scaling, tf.uint8)
            summaries.add(
                tf.summary.image('samples/%s' % ("label"), summary_label))
            first_clone_output=graph.get_tensor_by_name(('%s/%s:0'%(first_clone_scope,"semantic").strip('/')))
            prediction=tf.expand_dims(tf.argmax(first_clone_output,3),-1) # 先输出top3,然后整理成一列.
            summary_prediction=tf.cast(prediction*pixel_scaling,tf.uint8)
            summaries.add(tf.summary.image('samples/%s'% ("semantic"),summary_prediction))
        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name,loss))



        # åˆ›å»ºopt
        with tf.device(config.optimizer_device()):
            learning_rate = get_model_learning_rate(
                learning_policy,
                base_learning_rate,
                learning_rate_decay_step,
                learning_rate_decay_factor,
                training_number_of_steps,
                learning_power,
                slow_start_step,
                slow_start_learning_rate)
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            # add summary
            summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        startup_delay_steps = 15 * task  # startup_delay_steps=15#
        # losså’Œopt
        with tf.device(config.variables_device()):
            total_loss, grads_and_vars = model_deploy.optimize_clones(clones, optimizer)
            total_loss = tf.check_numerics(total_loss, 'total loss is inf or nan')
            # summary total loss
            summaries.add(tf.summary.scalar('total_loss', total_loss))
            # æ‹¿åˆ°æœ€åŽä¸€å±‚çš„vars
            if last_layers_contain_logits_only:
                last_layers = ['logits']
            else:
                last_layers = [
                    'logits',
                    'image_pooling',
                    'aspp',
                    'concat_projection',
                    'decoder',
                ]
            # å¦‚æžœæ¢¯åº¦éœ€è¦æŒ‰ç…§ä¸åŒçš„layerè‡ªå®šä¹‰å­˜åœ¨.
            grad_mul = get_model_gradient_multipliers(
                last_layers, last_layer_gradient_multiplier
            )

            if grad_mul:
                grads_and_vars = slim.learning.multiply_gradients(grads_and_vars, grad_mul)

            # åˆ›å»ºæ¢¯åº¦æ›´æ–°çš„æ“ä½œ
            grads_update = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            update_ops.append(grads_update)
            update_op = tf.group(*update_ops)  # æŠŠè¿™äº›opç»„åˆåœ¨ä¸€èµ·.
            with tf.control_dependencies([update_op]):  # BLOCK1_BUG:æ³¨æ„è¿™é‡Œæ˜¯()å¼•ç”¨.
                train_tensor = tf.identity(total_loss, name="train_op")

            session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            summaries|=set(tf.get_collection(tf.GraphKeys.SUMMARIES,first_clone_scope))
            summary_op=tf.summary.merge(list(summaries)) # 需要个list

            slim.learning.train(
                train_tensor,
                logdir=train_logdir,
                log_every_n_steps=log_steps,
                master=master,
                number_of_steps=training_number_of_steps,
                is_chief=(task == 0),
                session_config=session_config,
                startup_delay_steps=startup_delay_steps,
                init_fn=get_model_init_fn(
                    train_logdir,
                    tf_initial_checkpoint,
                    initialize_last_layer,
                    last_layers,
                    ignore_missing_vars=True,
                ),
                summary_op=summary_op,
                save_summaries_secs=save_summaries_secs,
                save_interval_secs=save_interval_secs
            )


def predict_labels(images,image_pyramid,fine_tune_batch_norm=False):
    """
    :param images: 输入图像数据.
    :param image_pyramid: 是否采用图像金字塔
    :return: 返回对应的预测的区域图.
    """
    tf.logging.info("[predict_labels] images:%s" % (images))
    tf.logging.info("[predict_labels] image_pyramid:%s" % (image_pyramid))
    tf.logging.info("[predict_labels] fine_tune_batch_norm:%s" % (fine_tune_batch_norm))

    _predict_output = multi_scale_logits(images=images,local_image_pyramid=image_pyramid,
                       weight_decay=0.0001,
                       is_training=True,
                       fine_tune_batch_norm=fine_tune_batch_norm)
    tf.logging.info("[predict_labels] _predict_output:%s"%(_predict_output))
    predictions={}
    for output in sorted(_predict_output):
        scales=_predict_output[output]
        tf.logging.info("[predict_labels] output:%s,sorted(_predict_output):%s, scales:%s,imagesize:%s" % (output, sorted(_predict_output),scales,tf.shape(images)[1:3]))
        logits=tf.image.resize_bilinear(scales["merged_logits"],size=tf.shape(images)[1:3],align_corners=True,)
        predictions[output]=tf.argmax(logits,3)
    tf.logging.info("[predict_labels] predictions:%s"%(predictions))
    return predictions



def eval():
    tf.logging.set_verbosity(tf.logging.INFO)
    eval_split='val' # val的数据集
    eval_crop_size=[513, 513]
    eval_batch_size=1
    min_resize_value=None
    max_resize_value=None
    resize_factor=None
    max_number_of_evaluations=0
    eval_scales=[1.0] # 注意这里是list类型
    dataset=get_dataset(dataset_name,eval_split,dataset_dir)
    tf.gfile.MakeDirs(eval_logdir)
    tf.logging.info('Evaluating on %s set', eval_split)
    with tf.Graph().as_default():
        samples=input_generator.get(dataset,
        eval_crop_size,
        eval_batch_size,
        min_resize_value=min_resize_value,
        max_resize_value=max_resize_value,
        resize_factor=resize_factor,
        dataset_split=eval_split,
        is_training=False,
        model_variant="xception_65")
        if tuple(eval_scales)==(1.0,):
            tf.logging.info("single scale")
            _predict_logits=predict_labels(samples["image"],local_image_pyramid,fine_tune_batch_norm=True)
        else:
            tf.logging.info("multi scales")

        _predict_logits=_predict_logits["semantic"]
        _predict_logits=tf.reshape(_predict_logits,shape=[-1])
        labels=tf.reshape(samples["label"],shape=[-1])
        tf.logging.info("[eval] labels:%s, _predict_logits:%s"%(labels,_predict_logits))
        weights=tf.to_float( tf.not_equal(labels,dataset.ignore_label)) # dataset.ignore_label==255
        tf.logging.info("[eval] weights:%s, dataset.ignore_label:%s, tf.no_equal(labels,dataset.ignore_label):%s" % (weights, dataset.ignore_label, tf.not_equal(labels,dataset.ignore_label)))
        tf.logging.info("[eval] tf.equal(labels,dataset.ignore_label):%s, tf.zeros_like(labels):%s" % (tf.equal(labels,dataset.ignore_label), tf.zeros_like(labels)))
        labels=tf.where(tf.equal(labels,dataset.ignore_label), tf.zeros_like(labels), labels)
        tf.logging.info("[eval] after tf.where, labels:%s " %(labels))

        # 计算miou
        metric_map={}
        metric_map["miou_1.0"]=tf.metrics.mean_iou(_predict_logits,labels,dataset.num_classes,weights=weights)
        tf.logging.info("[eval] metric_map:%s " % (metric_map))
        metric_vals,metrics_update=(tf.contrib.metrics.aggregate_metric_map(metric_map))
        tf.logging.info("[eval] metric_vals:%s ,metrics_update:%s" % (metric_vals,metrics_update))

        # summary
        for metric_name,metric_val in six.iteritems(metric_vals):
            slim.summaries.add_scalar_summary(metric_val,metric_name,print_summary=True)
        num_batches=int(math.ceil(dataset.num_samples/float(eval_batch_size)))
        tf.logging.info("[eval] num_batches:%s ,num_samples:%s, eval_batch_size:%s" % (num_batches, dataset.num_samples,eval_batch_size))

        num_eval_iterations=None
        if max_number_of_evaluations>0:
            num_eval_iterations=max_number_of_evaluations
        slim.evaluation.evaluation_loop(
            master=master,
            checkpoint_dir=checkpoint_dir,
            logdir=eval_logdir,
            num_evals=num_batches,
            eval_op=list(metrics_update.values()),
            max_number_of_evaluations=num_eval_iterations,
            eval_interval_secs= 60 * 5)




if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("NO action specified.")
        sys.exit()
    if sys.argv[1].startswith('--'):
        option = sys.argv[1][2:]
        if option == 'version':
            print("version 1.2 ")
        elif option == 'help':
            print("This program prints files to the standard output.\
                     Any number of files can be specified.\
                     Options include:\
                     --version : Prints the version number\
                     --train: train \
                     --eval:  eval \
                     --vis:   visual those picture's segmentation \
                     --help     : Display this help")
        elif option == 'train':
            print("start training")
            train()
        elif option == 'eval':
            print("start evaluating")
            eval()
        elif option == 'vis':
            print("start visualizing")
        else:
            raise ValueError("Unknow option.")
            sys.exit()