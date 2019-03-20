# tf print using var
import tensorflow as tf

## cls_ohem
cls_ohem = tf.placeholder(tf.float32)

inf_boxes = tf.placeholder(tf.float32)

### demo:
"""
from object_detection import tfprint
zeros_tsr = tf.zeros([2, 3]) ##为了调用tf.Print做的dummy.
tfprint.rpn_box_encodings = tf.Print(zeros_tsr,["rpn_box_encodings",tf.shape(rpn_box_encodings),rpn_box_encodings],summarize=64)

"""