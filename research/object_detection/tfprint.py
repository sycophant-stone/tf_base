# tf print using var
import tensorflow as tf

## ssd
ssd_fmap0 = tf.placeholder(tf.float32)
## rfcn
rfcn_roi = tf.placeholder(tf.float32)



### demo:
"""
from object_detection import tfprint
tfprint.rpn_box_encodings = tf.Print(rpn_box_encodings,["rpn_box_encodings",tf.shape(rpn_box_encodings),rpn_box_encodings],summarize=64)

"""