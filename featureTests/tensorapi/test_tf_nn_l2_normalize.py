import tensorflow as tf
input_data = tf.constant([[1.0,2,3],[4.0,5,6],[7.0,8,9]])

output = tf.nn.l2_normalize(input_data, dim = 0)
with tf.Session() as sess:
    print(sess.run(input_data))
    print(sess.run(output))
