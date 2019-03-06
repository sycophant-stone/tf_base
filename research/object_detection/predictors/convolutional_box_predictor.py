# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Convolutional Box Predictors with and without weight sharing."""
import tensorflow as tf
from object_detection.core import box_predictor
from object_detection.utils import static_shape
from object_detection import tfprint
from object_detection.utils import ops
slim = tf.contrib.slim

BOX_ENCODINGS = box_predictor.BOX_ENCODINGS
CLASS_PREDICTIONS_WITH_BACKGROUND = (
    box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND)
MASK_PREDICTIONS = box_predictor.MASK_PREDICTIONS


class _NoopVariableScope(object):
  """A dummy class that does not push any scope."""

  def __enter__(self):
    return None

  def __exit__(self, exc_type, exc_value, traceback):
    return False


class ConvolutionalBoxPredictor(box_predictor.BoxPredictor):
  """Convolutional Box Predictor.

  Optionally add an intermediate 1x1 convolutional layer after features and
  predict in parallel branches box_encodings and
  class_predictions_with_background.

  Currently this box predictor assumes that predictions are "shared" across
  classes --- that is each anchor makes box predictions which do not depend
  on class.
  """

  def __init__(self,
               is_training,
               num_classes,
               box_prediction_head,
               class_prediction_head,
               other_heads,
               conv_hyperparams_fn,
               num_layers_before_predictor,
               min_depth,
               max_depth):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      box_prediction_head: The head that predicts the boxes.
      class_prediction_head: The head that predicts the classes.
      other_heads: A dictionary mapping head names to convolutional
        head classes.
      conv_hyperparams_fn: A function to generate tf-slim arg_scope with
        hyperparameters for convolution ops.
      num_layers_before_predictor: Number of the additional conv layers before
        the predictor.
      min_depth: Minimum feature depth prior to predicting box encodings
        and class predictions.
      max_depth: Maximum feature depth prior to predicting box encodings
        and class predictions. If max_depth is set to 0, no additional
        feature map will be inserted before location and class predictions.

    Raises:
      ValueError: if min_depth > max_depth.
    """
    super(ConvolutionalBoxPredictor, self).__init__(is_training, num_classes)
    self._box_prediction_head = box_prediction_head
    self._class_prediction_head = class_prediction_head
    self._other_heads = other_heads
    self._conv_hyperparams_fn = conv_hyperparams_fn
    self._min_depth = min_depth
    self._max_depth = max_depth
    self._num_layers_before_predictor = num_layers_before_predictor
    print("[ConvolutionalBoxPredictor] _box_prediction_head",self._box_prediction_head)
    print("[ConvolutionalBoxPredictor] _class_prediction_head",self._class_prediction_head)
          
  @property
  def num_classes(self):
    return self._num_classes

  def _predict(self, image_features, num_predictions_per_location_list):
    """Computes encoded object locations and corresponding confidences.

    Args:
      image_features: A list of float tensors of shape [batch_size, height_i,
        width_i, channels_i] containing features for a batch of images.
      num_predictions_per_location_list: A list of integers representing the
        number of box predictions to be made per spatial location for each
        feature map.

    Returns:
      box_encodings: A list of float tensors of shape
        [batch_size, num_anchors_i, q, code_size] representing the location of
        the objects, where q is 1 or the number of classes. Each entry in the
        list corresponds to a feature map in the input `image_features` list.
      class_predictions_with_background: A list of float tensors of shape
        [batch_size, num_anchors_i, num_classes + 1] representing the class
        predictions for the proposals. Each entry in the list corresponds to a
        feature map in the input `image_features` list.
    """
    predictions = {
        BOX_ENCODINGS: [],
        CLASS_PREDICTIONS_WITH_BACKGROUND: [],
    }
    for head_name in self._other_heads.keys():
      predictions[head_name] = []
    # TODO(rathodv): Come up with a better way to generate scope names
    # in box predictor once we have time to retrain all models in the zoo.
    # The following lines create scope names to be backwards compatible with the
    # existing checkpoints.
    box_predictor_scopes = [_NoopVariableScope()]
    if len(image_features) > 1:
      box_predictor_scopes = [
          tf.variable_scope('BoxPredictor_{}'.format(i))
          for i in range(len(image_features))
      ]
    for idx, (image_feature,
         num_predictions_per_location, box_predictor_scope) in enumerate(zip(
             image_features, num_predictions_per_location_list,
             box_predictor_scopes)):
      net = image_feature
      with box_predictor_scope:
        with slim.arg_scope(self._conv_hyperparams_fn()):
          with slim.arg_scope([slim.dropout], is_training=self._is_training):
            # Add additional conv layers before the class predictor.
            features_depth = static_shape.get_depth(image_feature.get_shape())
            depth = max(min(features_depth, self._max_depth), self._min_depth)
            tf.logging.info('depth of additional conv before box predictor: {}'.
                            format(depth))
            if depth > 0 and self._num_layers_before_predictor > 0:
              for i in range(self._num_layers_before_predictor):
                net = slim.conv2d(
                    net,
                    depth, [1, 1],
                    reuse=tf.AUTO_REUSE,
                    scope='Conv2d_%d_1x1_%d' % (i, depth))
            sorted_keys = sorted(self._other_heads.keys())
            sorted_keys.append(BOX_ENCODINGS)
            sorted_keys.append(CLASS_PREDICTIONS_WITH_BACKGROUND)
            print("[ConvolutionalBoxPredictor._predict] sorted_keys",sorted_keys)
            for head_name in sorted_keys:
              print("[ConvolutionalBoxPredictor._predict] head_name",head_name)
              if head_name == BOX_ENCODINGS:
                head_obj = self._box_prediction_head
              elif head_name == CLASS_PREDICTIONS_WITH_BACKGROUND:
                head_obj = self._class_prediction_head
              else:
                head_obj = self._other_heads[head_name]
              print("[ConvolutionalBoxPredictor._predict] head_obj",head_obj)
              prediction = head_obj.predict(
                  features=image_feature,
                  num_predictions_per_location=num_predictions_per_location)
              predictions[head_name].append(prediction)
              '''
              if (idx==0):
                    image_feature0 = image_feature
                    if head_name == BOX_ENCODINGS:
                        prediction0reg = prediction
                        head_name0= head_name
                        tfprint.ssd_fmap0_reg = tf.Print(image_feature0,["ssd_fmap0,idx0,head_name", head_name0,tf.shape(image_feature0),tf.shape(prediction0reg)],summarize=10)
                    else:
                        prediction0cls = prediction
                        head_name0= head_name
                        tfprint.ssd_fmap0_cls = tf.Print(image_feature0,["ssd_fmap0,idx0,head_name", head_name0,tf.shape(image_feature0),tf.shape(prediction0cls)],summarize=10)
                    
              if (idx==1):
                    image_feature1 = image_feature
                    prediction1 = prediction
                    head_name1= head_name
                    tfprint.ssd_fmap1 = tf.Print(image_feature1,["ssd_fmap0,idx1,head_name", head_name1,tf.shape(image_feature1),tf.shape(prediction1)],summarize=10)
              if (idx==2):
                    image_feature2 = image_feature
                    prediction2 = prediction
                    head_name2= head_name
                    tfprint.ssd_fmap2 = tf.Print(image_feature2,["ssd_fmap0,idx2,head_name", head_name2,tf.shape(image_feature2),tf.shape(prediction2)],summarize=10)
              if (idx==3):
                    image_feature3 = image_feature
                    prediction3 = prediction
                    head_name3= head_name
                    tfprint.ssd_fmap3 = tf.Print(image_feature3,["ssd_fmap0,idx3,head_name", head_name3,tf.shape(image_feature3),tf.shape(prediction3)],summarize=10)
              if (idx==4):
                    image_feature4 = image_feature
                    prediction4 = prediction
                    head_name4= head_name
                    tfprint.ssd_fmap4 = tf.Print(image_feature4,["ssd_fmap0,idx4,head_name", head_name4,tf.shape(image_feature4),tf.shape(prediction4)],summarize=10)
              if (idx==5):
                    image_feature5 = image_feature
                    prediction5 = prediction
                    head_name5= head_name
                    tfprint.ssd_fmap5 = tf.Print(image_feature5,["ssd_fmap0,idx5,head_name", head_name5,tf.shape(image_feature5),tf.shape(prediction5)],summarize=10)
             '''
                    
              #if(idx==0):
                ## add rfcn roi
                #tfprint.ssd_fmap0 = tf.Print(image_feature,["ssd_fmap0",tf.shape(image_feature),tf.shape(predictions['box_encodings']),tf.shape(predictions['class_predictions_with_background'])],summarize=64)
              
              if(idx==0):
              ### add roi for 1st feature maps
                  net_roi = image_feature
                  proposal_boxes = predictions[BOX_ENCODINGS]
                  #th slim.arg_scope(self._conv_hyperparams_fn()):
                  
                  _depth = 1024
                  net_roi = slim.conv2d(net_roi, _depth, [1, 1],reuse=tf.AUTO_REUSE, scope='reduce_depth_roi')
                  
                  # Location predictions.
                  _num_spatial_bins = [3,3]
                  _num_classes = 20
                  _box_code_size = 4
                  _crop_size = [18, 18]
                  batch_size = tf.shape(proposal_boxes[0])[0]
                  num_boxes = tf.shape(proposal_boxes[0])[1]
                  item2 = tf.shape(proposal_boxes[0])[2]
                  item3 = tf.shape(proposal_boxes[0])[3]
                  # 这部分的结论已有. 看起来是正确的. net_roi是[24 19 19 1024]
                  #tfprint.ssd_debug0 = tf.Print(net_roi,["reduce depth roi, img, dpt, out; batch_size,num_boxes",tf.shape(image_feature),_depth,tf.shape(net_roi),batch_size,num_boxes],summarize=8)
                  #tfprint.ssd_debug0 = tf.Print(net_roi,["proposal_boxes' shape",batch_size,num_boxes,item2,item3],summarize=8)
                  
                  location_feature_map_depth = (_num_spatial_bins[0] *
                                            _num_spatial_bins[1] *
                                            _num_classes *
                                            _box_code_size)
                  location_feature_map = slim.conv2d(net_roi, location_feature_map_depth,
                                                [1, 1], activation_fn=None,
                                                     reuse=tf.AUTO_REUSE,
                                                scope='refined_locations_roi')
                  ##tf.shape(location_feature_map)
                  proposal_boxes = tf.squeeze(proposal_boxes[0],axis=[2]) #把[24 1083 1 4]的dim0,dim3的"1"挤掉.因为batch_position_sensitive_crop_regions
                  
                  box_encodings = ops.batch_position_sensitive_crop_regions(
                    location_feature_map,
                    boxes=proposal_boxes,
                    crop_size=_crop_size,
                    num_spatial_bins=_num_spatial_bins,
                    global_pool=True)
                  
                  box_encodings = tf.squeeze(box_encodings, squeeze_dims=[2, 3]) #pos reg[24, 1083 1 1 80],带有batch的.
                  tfprint.pos_sen = tf.Print(image_feature,["squeezed box",tf.shape(box_encodings)],summarize=8)
                  '''注意,如果tf.Print后面接的第一个参数是tensor,如果这个tensor尺寸太大,tf.print会打印它的值.这会导致GPU memory overflow.
                     建议把tensor设置成一个小值,我们重点看第二列的shape值.'''
                  ##box_encodings = tf.reshape(box_encodings,[batch_size * num_boxes, 1, _num_classes,_box_code_size])
                  #box_encodings = tf.reshape(box_encodings,[batch_size , num_boxes, _num_classes,_box_code_size])
                  
                  # Class predictions.
                  ''' 先只看reg
                  total_classes = _num_classes + 1  # Account for background class.
                  class_feature_map_depth = (_num_spatial_bins[0] *
                                        _num_spatial_bins[1]*
                                        total_classes)
                  class_feature_map = slim.conv2d(net_roi, class_feature_map_depth, [1, 1],
                                                activation_fn=None,
                                                   reuse=tf.AUTO_REUSE,
                                                scope='class_predictions_roi')
                  
                  class_predictions_with_background = (
                    ops.batch_position_sensitive_crop_regions(
                        class_feature_map,
                        boxes=proposal_boxes,
                        crop_size=_crop_size,
                        num_spatial_bins=_num_spatial_bins,
                        global_pool=True))
                  class_predictions_with_background = tf.squeeze(
                    class_predictions_with_background, squeeze_dims=[2, 3])
                  class_predictions_with_background = tf.reshape(
                    class_predictions_with_background,
                    [batch_size * num_boxes, 1, total_classes])
                 ''' 
                  #tfprint.rfcn_roi = tf.Print(class_feature_map,["rfcn roi, cls and reg",tf.shape(class_feature_map),tf.shape(class_predictions_with_background),tf.shape(location_feature_map),tf.shape(box_encodings)],summarize=64)
                  ## box_encodings不能打印.
                  
                  
                  ## change dims to match the ssd' outputs.
                  # 这个是没必要,尺寸不对.rshp_box_encodings = slim.conv2d(box_encodings , 1083, [1, 1], reuse=tf.AUTO_REUSE, scope='RoiRegPostReshape') #[24 1083 1 4]
                  #class_predictions_with_background = tf.expand_dims(class_predictions_with_background,axis=2)#在dim1上添加一个维度.
                  #rshp_class_predictions_with_background = slim.conv2d(class_predictions_with_background , 1083 , [1, 1],  reuse=tf.AUTO_REUSE,scope='RoiClsPostReshape') #[24 1083 21] 这里不对. 1083个输出是不对的.
                  
                  #tfprint.ssd_debug0 = tf.Print(box_encodings,["box_encodings,rshp_box_encodings,class_predictions_with_background,reshp___",tf.shape(box_encodings),tf.shape(rshp_box_encodings),tf.shape(class_predictions_with_background),tf.shape(rshp_class_predictions_with_background)],summarize=4)
                
                  ## add to ssd's prediction outputs
                  ## dim isn't equal, remove to debug.
                  #predictions['box_encodings'].append(rshp_box_encodings)
                  #predictions['class_predictions_with_background'].append(rshp_class_predictions_with_background)
                  
              ### end roi for 1st maps
              
    return predictions


# TODO(rathodv): Replace with slim.arg_scope_func_key once its available
# externally.
def _arg_scope_func_key(op):
  """Returns a key that can be used to index arg_scope dictionary."""
  return getattr(op, '_key_op', str(op))


# TODO(rathodv): Merge the implementation with ConvolutionalBoxPredictor above
# since they are very similar.
class WeightSharedConvolutionalBoxPredictor(box_predictor.BoxPredictor):
  """Convolutional Box Predictor with weight sharing.

  Defines the box predictor as defined in
  https://arxiv.org/abs/1708.02002. This class differs from
  ConvolutionalBoxPredictor in that it shares weights and biases while
  predicting from different feature maps. However, batch_norm parameters are not
  shared because the statistics of the activations vary among the different
  feature maps.

  Also note that separate multi-layer towers are constructed for the box
  encoding and class predictors respectively.
  """

  def __init__(self,
               is_training,
               num_classes,
               box_prediction_head,
               class_prediction_head,
               other_heads,
               conv_hyperparams_fn,
               depth,
               num_layers_before_predictor,
               kernel_size=3,
               apply_batch_norm=False,
               share_prediction_tower=False):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      box_prediction_head: The head that predicts the boxes.
      class_prediction_head: The head that predicts the classes.
      other_heads: A dictionary mapping head names to convolutional
        head classes.
      conv_hyperparams_fn: A function to generate tf-slim arg_scope with
        hyperparameters for convolution ops.
      depth: depth of conv layers.
      num_layers_before_predictor: Number of the additional conv layers before
        the predictor.
      kernel_size: Size of final convolution kernel.
      apply_batch_norm: Whether to apply batch normalization to conv layers in
        this predictor.
      share_prediction_tower: Whether to share the multi-layer tower between box
        prediction and class prediction heads.
    """
    super(WeightSharedConvolutionalBoxPredictor, self).__init__(is_training,
                                                                num_classes)
    self._box_prediction_head = box_prediction_head
    self._class_prediction_head = class_prediction_head
    self._other_heads = other_heads
    self._conv_hyperparams_fn = conv_hyperparams_fn
    self._depth = depth
    self._num_layers_before_predictor = num_layers_before_predictor
    self._kernel_size = kernel_size
    self._apply_batch_norm = apply_batch_norm
    self._share_prediction_tower = share_prediction_tower
    print("[ConvolutionalBoxPredictor] _predict_head",self._predict_head)
  @property
  def num_classes(self):
    return self._num_classes

  def _insert_additional_projection_layer(self, image_feature,
                                          inserted_layer_counter,
                                          target_channel):
    if inserted_layer_counter < 0:
      return image_feature, inserted_layer_counter
    image_feature = slim.conv2d(
        image_feature,
        target_channel, [1, 1],
        stride=1,
        padding='SAME',
        activation_fn=None,
        normalizer_fn=(tf.identity if self._apply_batch_norm else None),
        scope='ProjectionLayer/conv2d_{}'.format(
            inserted_layer_counter))
    if self._apply_batch_norm:
      image_feature = slim.batch_norm(
          image_feature,
          scope='ProjectionLayer/conv2d_{}/BatchNorm'.format(
              inserted_layer_counter))
    inserted_layer_counter += 1
    return image_feature, inserted_layer_counter

  def _compute_base_tower(self, tower_name_scope, image_feature, feature_index,
                          has_different_feature_channels, target_channel,
                          inserted_layer_counter):
    net = image_feature
    for i in range(self._num_layers_before_predictor):
      net = slim.conv2d(
          net,
          self._depth, [self._kernel_size, self._kernel_size],
          stride=1,
          padding='SAME',
          activation_fn=None,
          normalizer_fn=(tf.identity if self._apply_batch_norm else None),
          scope='{}/conv2d_{}'.format(tower_name_scope, i))
      if self._apply_batch_norm:
        net = slim.batch_norm(
            net,
            scope='{}/conv2d_{}/BatchNorm/feature_{}'.
            format(tower_name_scope, i, feature_index))
      net = tf.nn.relu6(net)
    return net

  def _predict_head(self, head_name, head_obj, image_feature, box_tower_feature,
                    feature_index, has_different_feature_channels,
                    target_channel, inserted_layer_counter,
                    num_predictions_per_location):
    if head_name == CLASS_PREDICTIONS_WITH_BACKGROUND:
      tower_name_scope = 'ClassPredictionTower'
    elif head_name == MASK_PREDICTIONS:
      tower_name_scope = 'MaskPredictionTower'
    else:
      raise ValueError('Unknown head')
    if self._share_prediction_tower:
      head_tower_feature = box_tower_feature
    else:
      head_tower_feature = self._compute_base_tower(
          tower_name_scope=tower_name_scope,
          image_feature=image_feature,
          feature_index=feature_index,
          has_different_feature_channels=has_different_feature_channels,
          target_channel=target_channel,
          inserted_layer_counter=inserted_layer_counter)
    return head_obj.predict(
        features=head_tower_feature,
        num_predictions_per_location=num_predictions_per_location)

  def _predict(self, image_features, num_predictions_per_location_list):
    """Computes encoded object locations and corresponding confidences.

    Args:
      image_features: A list of float tensors of shape [batch_size, height_i,
        width_i, channels] containing features for a batch of images. Note that
        when not all tensors in the list have the same number of channels, an
        additional projection layer will be added on top the tensor to generate
        feature map with number of channels consitent with the majority.
      num_predictions_per_location_list: A list of integers representing the
        number of box predictions to be made per spatial location for each
        feature map. Note that all values must be the same since the weights are
        shared.

    Returns:
      A dictionary containing:
        box_encodings: A list of float tensors of shape
          [batch_size, num_anchors_i, code_size] representing the location of
          the objects. Each entry in the list corresponds to a feature map in
          the input `image_features` list.
        class_predictions_with_background: A list of float tensors of shape
          [batch_size, num_anchors_i, num_classes + 1] representing the class
          predictions for the proposals. Each entry in the list corresponds to a
          feature map in the input `image_features` list.
        (optional) mask_predictions: A list of float tensors of shape
          [batch_size, num_anchord_i, num_classes, mask_height, mask_width].


    Raises:
      ValueError: If the image feature maps do not have the same number of
        channels or if the num predictions per locations is differs between the
        feature maps.
    """
    if len(set(num_predictions_per_location_list)) > 1:
      raise ValueError('num predictions per location must be same for all'
                       'feature maps, found: {}'.format(
                           num_predictions_per_location_list))
    feature_channels = [
        image_feature.shape[3].value for image_feature in image_features
    ]
    has_different_feature_channels = len(set(feature_channels)) > 1
    if has_different_feature_channels:
      inserted_layer_counter = 0
      target_channel = max(set(feature_channels), key=feature_channels.count)
      tf.logging.info('Not all feature maps have the same number of '
                      'channels, found: {}, addition project layers '
                      'to bring all feature maps to uniform channels '
                      'of {}'.format(feature_channels, target_channel))
    else:
      # Place holder variables if has_different_feature_channels is False.
      target_channel = -1
      inserted_layer_counter = -1
    predictions = {
        BOX_ENCODINGS: [],
        CLASS_PREDICTIONS_WITH_BACKGROUND: [],
    }
    for head_name in self._other_heads.keys():
      predictions[head_name] = []
    for feature_index, (image_feature,
                        num_predictions_per_location) in enumerate(
                            zip(image_features,
                                num_predictions_per_location_list)):
      with tf.variable_scope('WeightSharedConvolutionalBoxPredictor',
                             reuse=tf.AUTO_REUSE):
        with slim.arg_scope(self._conv_hyperparams_fn()):
          (image_feature,
           inserted_layer_counter) = self._insert_additional_projection_layer(
               image_feature, inserted_layer_counter, target_channel)
          if self._share_prediction_tower:
            box_tower_scope = 'PredictionTower'
          else:
            box_tower_scope = 'BoxPredictionTower'
          box_tower_feature = self._compute_base_tower(
              tower_name_scope=box_tower_scope,
              image_feature=image_feature,
              feature_index=feature_index,
              has_different_feature_channels=has_different_feature_channels,
              target_channel=target_channel,
              inserted_layer_counter=inserted_layer_counter)
          box_encodings = self._box_prediction_head.predict(
              features=box_tower_feature,
              num_predictions_per_location=num_predictions_per_location)
          predictions[BOX_ENCODINGS].append(box_encodings)
          sorted_keys = sorted(self._other_heads.keys())
          sorted_keys.append(CLASS_PREDICTIONS_WITH_BACKGROUND)
          for head_name in sorted_keys:
            if head_name == CLASS_PREDICTIONS_WITH_BACKGROUND:
              head_obj = self._class_prediction_head
            else:
              head_obj = self._other_heads[head_name]
            prediction = self._predict_head(
                head_name=head_name,
                head_obj=head_obj,
                image_feature=image_feature,
                box_tower_feature=box_tower_feature,
                feature_index=feature_index,
                has_different_feature_channels=has_different_feature_channels,
                target_channel=target_channel,
                inserted_layer_counter=inserted_layer_counter,
                num_predictions_per_location=num_predictions_per_location)
            predictions[head_name].append(prediction)
    return predictions
