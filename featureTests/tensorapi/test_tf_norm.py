import tensorflow as tf
input_data = tf.constant([[1.0,2,3],[4.0,5,6],[7.0,8,9]])

## 计算的是矩阵的frobenius范数. (https://www.zhihu.com/question/22475774)
## 当axis=0时, 每一列的元素平方和,然后开方 1+4^2+7^2 = 66; np.sqrt(66)= 8.124039
output = tf.norm(input_data, axis=0, keepdims=True)
with tf.Session() as sess:
    print("input_data shape:", sess.run(tf.shape(input_data)))
    print(sess.run(input_data))
    print("output shape", sess.run(tf.shape(output)))
    print(sess.run(output))
    
'''输出:
input_data shape: [3 3]
[[1. 2. 3.]
 [4. 5. 6.]
 [7. 8. 9.]]
output shape [1 3]
[[ 8.124039  9.643651 11.224972]]
'''
