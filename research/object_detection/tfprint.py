# tf print using var
import tensorflow as tf

## ssd
ssd_fmap0 = tf.placeholder(tf.float32)
ssd_fmap0_reg = tf.placeholder(tf.float32)
ssd_fmap0_cls = tf.placeholder(tf.float32)

ssd_fmap1 = tf.placeholder(tf.float32)
ssd_fmap2 = tf.placeholder(tf.float32)
ssd_fmap3 = tf.placeholder(tf.float32)
ssd_fmap4 = tf.placeholder(tf.float32)
ssd_fmap5 = tf.placeholder(tf.float32)

ssd_debug0 = tf.placeholder(tf.float32)


## rfcn
rfcn_roi = tf.placeholder(tf.float32)
pos_sen = tf.placeholder(tf.float32)

### demo:
"""
from object_detection import tfprint
tfprint.rpn_box_encodings = tf.Print(rpn_box_encodings,["rpn_box_encodings",tf.shape(rpn_box_encodings),rpn_box_encodings],summarize=64)

"""