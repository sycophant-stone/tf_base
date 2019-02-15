# tf print using var
import tensorflow as tf
tfp_similarity_matrix = tf.placeholder(tf.float32)
tfp_match_results = tf.placeholder(tf.float32)
argmax_row_range = tf.placeholder(tf.float32)
argmax_col_range = tf.placeholder(tf.float32)
forced_matches_values = tf.placeholder(tf.float32)
keep_matches_ids = tf.placeholder(tf.float32)
keep_matches_values = tf.placeholder(tf.float32)
cls_targets = tf.placeholder(tf.float32)
reg_targets = tf.placeholder(tf.float32)
anchors= tf.placeholder(tf.float32)
groundtruth_boxes= tf.placeholder(tf.float32)
matches_raw = tf.placeholder(tf.float32)
matches_thresh = tf.placeholder(tf.float32)
## target_assigner.py
groundtruth_label = tf.placeholder(tf.float32)
target_assign_param = tf.placeholder(tf.float32)

## core/balanced_positive_negative_sampler.py
indicator = tf.placeholder(tf.bool) # not fetchable
labels = tf.placeholder(tf.bool)    # not fetchable

## faster rcnn meta arch 
rpn_box_encodings = tf.placeholder(tf.float32)
rpn_objectness_predictions_with_background = tf.placeholder(tf.float32)
f_anchors = tf.placeholder(tf.float32)
image_shape = tf.placeholder(tf.float32)
batch_shapes = tf.placeholder(tf.float32)
_postprocess_rpn = tf.placeholder(tf.float32)
refined_box_encodings = tf.placeholder(tf.float32)
_postprocess_rpn1 = tf.placeholder(tf.float32)

## anchor_generator
anchor_generator_param = tf.placeholder(tf.float32)

## faster_rcnn_box_coder.py
box_decoder_param = tf.placeholder(tf.float32)


### demo:
"""
from object_detection import tfprint
tfprint.rpn_box_encodings = tf.Print(rpn_box_encodings,["rpn_box_encodings",tf.shape(rpn_box_encodings),rpn_box_encodings],summarize=64)

"""