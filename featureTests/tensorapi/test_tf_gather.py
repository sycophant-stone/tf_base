import tensorflow as tf

temp = tf.range(0,10)*10 + tf.constant(1,shape=[10])
temp2 = tf.gather(temp,[1,5,9])
temp3 = tf.gather(temp,[[1,5,9],[2,6,8]])
temp4 = tf.gather(temp,[[1],[2],[6]])

with tf.Session() as sess:
    print(sess.run(temp))
    print(sess.run(temp2))
    print(sess.run(temp3))
    print(sess.run(temp4))
