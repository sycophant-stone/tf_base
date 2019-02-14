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
	
def test_tf_slice():
	t1 = tf.constant(value=[1,2,3,4,5,6,7,8,9])
	t2 = tf.slice(t1,[0],tf.reshape(9,[1]))
	t3 = tf.slice(t1,[0],tf.reshape(3,[1]))
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print(sess.run(t1))
		print(sess.run(tf.shape(t1)))
		print(sess.run(t2))
		print(sess.run(t3))

def test_tf_dynamic_stitch():
	with tf.Session() as sess:
		indices = [6, [4, 1], [[5, 2], [0, 3]]]  
		data = [[61, 62], [[41, 42], [11, 12]], [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]]
		output = tf.dynamic_stitch(indices, data)
		print(sess.run(output))


def test_tf_range():
    x1 = tf.range(8)
    with tf.Session() as sess:
        print(sess.run(x1))

def test_tf_stack():
    x1 = tf.constant(value=[[1,2,3],[4,5,6],[7,8,9]])
    x2 = tf.constant(value=[[11,12,13],[14,15,16],[17,18,19]])
    b1 = tf.stack([x1,x2],axis=0)
    b2 = tf.stack([x1,x2],axis=1)
    b3 = tf.stack([x1,x2],axis=2)
    with tf.Session() as sess:
        print(sess.run(tf.shape(x1)))
        print(sess.run(tf.shape(x2)))
        print(sess.run(tf.shape(b1)))
        print(sess.run(tf.shape(b2)))
        print(sess.run(tf.shape(b3)))
        print(sess.run(b1))
        print(sess.run(b2))
        print(sess.run(b3))

def test_tf_squeeze():
    x1 = tf.constant(value=[1, 2, 1, 3, 1, 1])
    b1 = tf.squeeze([x1],axis=0)
    c1 = tf.squeeze(x1)
    #b2 = tf.squeeze([x1],axis=1)
    #b3 = tf.squeeze(x1,axis=2)
    with tf.Session() as sess:
        print(sess.run(tf.shape(b1)))
        print(sess.run(tf.shape(c1)))
        #print(sess.run(tf.shape(b2)))
        print(sess.run(b1))
        print(sess.run(c1))
        #print(sess.run(b2))

def test_shape():
    x1 = tf.constant(value=[[1,2,3],[4,5,6],[7,8,9]])
    static_shape = x1.shape.as_list()
    dynamic_shape = tf.shape(x1)
    with tf.Session() as sess:
        print(static_shape)
        print(sess.run(dynamic_shape))

def test_tfnnsoftmax():
    A = [1.0,2.0,3.0,4.0,5.0,6.0]
    with tf.Session() as sess:
        print(sess.run(tf.nn.softmax(A)))
  
if __name__ == '__main__':
	#test_tf_gather()
	#test_tf_reshape()
	#test_tf_where()
	#test_tf_slice()
	#test_tf_dynamic_stitch()
    #test_tf_range()
    #test_tf_stack()
    #test_tf_squeeze()
    #test_shape()
    test_tfnnsoftmax()
    