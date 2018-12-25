import tensorflow as tf

def test_tf_gather():
	x1 = tf.constant(value=[[1,2,3],[4,5,6],[7,8,9]])
	t1 = tf.constant(value=[1,2,3,4,5,6,7,8,9])
	x2 = tf.gather(x1,[1,2])
	x3 = tf.gather(t1,5)
	x4 = tf.gather(x1,2)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print(sess.run(x1))
		print(sess.run(x2))
		print(sess.run(x3))
		print(sess.run(x4))
		
	
def test_tf_reshape():

	x1 = tf.constant(value=[[1,2,3],[4,5,6],[7,8,9]])
	x2 = tf.reshape(x1, [-1])
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print(sess.run(x1))
		print(sess.run(x2))
	
	
def test_tf_where():
	import numpy as np
	sess=tf.Session()
	
	a=np.array([[1,0,0],[0,1,1]])
	a1=np.array([[3,2,3],[4,5,6]])
	
	a2=np.array([-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1, 3 ,-1 ,-1 ,-1 ,-1 ,-1, 2 ,-1 ,3 ,3 ,-1, 3 ,-1 ,3 ,-1 ,-1 ,-1 ,3, 3 ,-1 ,-1, 3 ,-1 ,3 ,-1 ,-1 ,-1 ,-1, 2 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1, 3 ,-1 ,-1, 3])
	
	print(sess.run(tf.equal(a,1)))
	print(sess.run(tf.where(tf.equal(a,1))))
	print(sess.run(tf.where(tf.greater(a2, -1))))
	
if __name__ == '__main__':
	#test_tf_gather()
	#test_tf_reshape()
	test_tf_where()