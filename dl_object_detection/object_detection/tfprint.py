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
anchors= tf.placeholder(tf.float32)
groundtruth_boxes= tf.placeholder(tf.float32)
matches_raw = tf.placeholder(tf.float32)
matches_thresh = tf.placeholder(tf.float32)