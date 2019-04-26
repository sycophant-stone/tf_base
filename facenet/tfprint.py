# tf print using var
import tensorflow as tf

## center loss
center_loss = tf.placeholder(tf.float32)
asoftmax_loss = tf.placeholder(tf.float32)
### demo:
"""
import tfprint
zeros_tsr = tf.zeros([2, 3]) ##为了调用tf.Print做的dummy.
tfprint.rpn_box_encodings = tf.Print(zeros_tsr,["rpn_box_encodings",tf.shape(rpn_box_encodings),rpn_box_encodings],summarize=64)

"""