'''
   This program is for SegNet.
'''

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import os, sys
import numpy as np
import math
from datetime import datetime
import time
from PIL import Image
from math import ceil
from tensorflow.python.ops import gen_nn_ops
# modules
#from Utils import _variable_with_weight_decay, _variable_on_cpu, _add_loss_summaries, _activation_summary, print_hist_summery, get_hist, per_class_acc, writeImage
#from Inputs import *

'''-------'''
# parma presetting
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480
IMAGE_DEPTH = 3
NUM_CLASSES=12

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

INITIAL_LEARNING_RATE = 0.001 

'''------------------------------------------------------------------'''
##input

def get_filename_list(filename):
	# 拿到数据集
	fd =open(filename)
	image_filenames=[]
	label_filenames=[]
	for line in fd:
		line=line.strip().split(" ")
		image_filenames.append(line[0])
		label_filenames.append(line[1])
	
	return image_filenames,label_filenames

def CamVidInput(inputdatafilename,inputlabelfilename,batch_size):
    """
    inputdatafilename: 输入data
    inputlabelfilename: 输入y
    batch_size: 一次计算的量
    读到文件名队列中,然后读出图片.reshape
    """
    # 把数据集的文件名也保存成张量模式. 须知tensorflow就是对张量的操作.
    images=ops.convert_to_tensor(inputdatafilename,dtype=dtypes.string)
    labels=ops.convert_to_tensor(inputlabelfilename,dtype=dtypes.string)
    print("images:",images)
    print("labels:",labels)
    # 把这些数据集名字读入到内存中.
    filename_queue=tf.train.slice_input_producer([images,labels],shuffle=True)
    image_val=tf.read_file(filename_queue[0])
    label_val=tf.read_file(filename_queue[1])
    
    # 数据集是png图片,解析.
    image_bytes=tf.image.decode_png(image_val)
    label_bytes=tf.image.decode_png(label_val)
    
    # 把读到的png图片做reshape
    image=tf.reshape(image_bytes,(IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH))
    label=tf.reshape(label_bytes,(IMAGE_HEIGHT,IMAGE_WIDTH,1))
    
    return image,label
    
'''------------------------------------------------------------------'''
# Networks

def orthogonal_initializer(scale=1.1): # 正交
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    def _initializer(shape,dtype=tf.float32,partition_info=None):
        print("[%s]:  shape[0]:%s, shape[1]:%s"%("orthogonal_initializer",shape[0],shape[1]))
        flat_shape=(shape[0],np.prod(shape[1:])) # 这里shape[1:] 不是shape[1,:],shape本身并不是多维的
        a=np.random.normal(0.0,1.0,flat_shape)
        u,_,v=np.linalg.svd(a,full_matrices=False)
        q=u if u.shape==flat_shape else v
        q=q.reshape(shape)
        return tf.constant(scale*q[:shape[0],:shape[1]],dtype=tf.float32) # 0:shape[0]
    return _initializer
        

def helper_variable_on_cpu(name,shape,initializer):
    with tf.device('/gpu:0'):
        var=tf.get_variable(name,shape,initializer=initializer)
    return var

def helper_variable_with_weight_decay(name,shape,initializer,wd):
    var=helper_variable_on_cpu(name,shape,initializer)
    if wd is not None:
        weight_decay=tf.multiply(tf.nn.l2_loss(var),wd,name="weight_loss")
        tf.add_to_collection("losses",weight_decay)
    return var

def helper_add_loss_summaries(total_loss):
    loss_average=tf.train.ExponentialMovingAverage(0.9,name='avg')
    losses=tf.get_collection('losses')
    loss_average_op=loss_average.apply(losses+[total_loss])
    
    """ 
    summary
    for l in losses+[total_loss]:
        
    """
    return loss_average_op
    
def batch_norm_layer(inputI,is_trainning,scope):
    return tf.cond(is_trainning,
                   lambda: tf.contrib.layers.batch_norm(inputI,is_trainning=True,center=False,updates_collections=None,scope=scope+"_bn"),
                   lambda: tf.contrib.layers.batch_norm(inputI,is_trainning=False,center=False,updates_collections=None,scope=scope+"_bn",reuse=True))

def conv_layer_with_bn(inputI,shape,train_phase,activation=True,name=None):
    in_ch=shape[2]
    out_ch=shape[3]
    k_size=shape[0]
    with tf.variable_scope(name) as scope:
        kernel=helper_variable_with_weight_decay("ort_weights",shape=shape,initializer=orthogonal_initializer(),wd=None) # 注意此处orthogonal_initializer一定是该函数的返回值
        conv=tf.nn.conv2d(inputI,kernel,[1,1,1,1],padding='SAME')# 1x1 kernel size, 1 batch , 1 chn
        biases=helper_variable_on_cpu('biases',[out_ch],tf.constant_initializer(0.0))
        bias= tf.nn.bias_add(conv,biases)
        if activation is True:
            conv_out=tf.nn.relu(batch_norm_layer(bias,train_phase,scope.name))
        else:
            conv_out=atch_norm_layer(bias,train_phase,scope.name)
    return conv_out

        

def get_decode_filter(f_shape):
    """
    reference: https://github.com/MarvinTeichmann/tensorflow-fcn
    """
    width=f_shape[0]
    height=f_shape[0]
    f=ceil(width/2.0)
    c=(2*f-1-f%2)/(2.0*f)
    bilinear=np.zeros([f_shape[0],f_shape[1]])
    # 双线性插值
    for x in range(width):
        for y in range(height):
            value =(1-np.abs(x/f-c))*(1-np.abs(y/f-c))
            bilinear[x,y]=value
    weights=np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:,:,i,i]=bilinear
    init=tf.constant_initializer(value=weights,dtype=tf.float32)
    return tf.get_variable(name="up_filter",initializer=init,shape=weights.shape)
        
        
def decode_layer(inputI,f_shape,output_shape,stride=2,name=None):
    # 这个shape的格式: kernel_w,kernel_h,batch,chn
    sess_temp=tf.global_variables_initializer()
    strides=[1,stride,stride,1]
    with tf.variable_scope(name):
        weights=get_decode_filter(f_shape)
        deconv=tf.nn.conv2d_transpose(inputI,weights,output_shape,stride=stride,padding='SAME')
    
    return deconv

def msra_initializer(k1,d1):
    """
    k1: kernel size
    d1: filter number
    """
    stddev=math.sqrt(2.0/(k1**2*d1))
    return tf.truncated_normal_initializer(stddev=stddev)

def weight_loss(logits,labels,num_classes,head=None):
    """
    median-frequency re-weighting
    """
    with tf.name_scope('loss'):
        logits=tf.reshape(logits,(-1,num_classes))
        epsilon=tf.constant(value=1e-10)
        logits=logits+epsilon
        label_flat=tf.reshape(labels,(-1,1))
        labels=tf.reshape(tf.one_hot(label_flat,depth=num_classes),(-1,num_classes))
        softmax=tf.nn.softmax(logits)
        cross_entropy=tf.reduce_sum(tf.multiply(labels*tf.log(softmax+epsilon),head),axis=[1])
        cross_entropy_mean=tf.reduce_mean(cross_entropy,name='cross_entropy')
        tf.add_to_collection('losses',cross_entropy_mean)
        loss=tf.add_n(tf.get_collection('losses'),name='total_loss')
    
    return loss

def cal_loss(logits,labels):
    loss_weight = np.array([
      0.2595,
      0.1826,
      4.5640,
      0.1417,
      0.9051,
      0.3826,
      9.6446,
      1.8418,
      0.6823,
      6.2478,
      7.3614,
      1.0974]) # class 0~11
    labels=tf.cast(labels,tf.int32)
    return weight_loss(logits,labels,NUM_CLASSES,head=loss_weight)
        

    

def inference(images,labels,batch_size,phase_train):
    """
    images,labels: 
                读入的数据集
    batch_size:
                按批次的做训练
    phase_train:
                是个Bool类型.
    """
    # local response normalization
    norm1=tf.nn.lrn(images,depth_radius=5,bias=1.0,alpha=0.0001,beta=0.75,name="norm1")
    # conv1
    conv1=conv_layer_with_bn(norm1,[7,7,images.get_shape().as_list()[3],64],phase_train,name="conv1")
    # pool1
    pool1,pool1_indices=tf.nn.max_pool_with_argmax(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool1')
    
    # conv2
    conv2=conv_layer_with_bn(pool1,[7,7,64,64],phase_train,name='conv2')
    # pool2
    pool2,pool2_indices=tf.nn.max_pool_with_argmax(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool2')
    
    # conv3
    conv3=conv_layer_with_bn(pool2,[7,7,64,64],phase_train,name='conv3')
    # pool3
    pool3,pool3_indices=tf.nn.max_pool_with_argmax(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool3')
    
    # conv4
    conv4=conv_layer_with_bn(pool3,[7,7,64,64],phase_train,name='conv4')
    # pool3
    pool4,pool4_indices=tf.nn.max_pool_with_argmax(conv4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool4')
    """ end of encoder"""
    
    """upsampling"""
    
    # upsamping4
    upsample4=decode_layer(pool4,[2,2,64,64],[batch_size,45,60,64],2,"up4")
    # decode4
    conv_decode4=conv_layer_with_bn(upsample4,[7,7,64,64],False,"conv_decode4")
    
    # upsamping3
    upsample3=decode_layer(conv_decode4,[2,2,64,64],[batch_size,90,120,64],2,"up3")
    # decode3
    conv_decode3=conv_layer_with_bn(upsample3,[7,7,64,64],False,"conv_decode3")
    
    # upsamping2
    upsample2=decode_layer(conv_decode3,[2,2,64,64],[batch_size,180,240,64],2,"up2")
    # decode2
    conv_decode2=conv_layer_with_bn(upsample2,[7,7,64,64],False,"conv_decode2")
    
    # upsampling1
    upsample1=decode_layer(conv_decode2,[2,2,64,64],[batch_size,360,480,64],2,"up1")
    # decode2
    conv_decode1=conv_layer_with_bn(upsample1,[7,7,64,64],False,"conv_decode1")
    
    """ end of decode"""
    
    
    """ classify"""
    
    with tf.variable_scope('conv_classifier') as scope:
        kernel=helper_variable_with_weight_decay("weights",shape=[1,1,64,NUM_CLASSES],initializer=msra_initializer(1,64),
                                                wd=0.0005)
        conv=tf.nn.conv2d(conv_decode1,kernel,[1,1,1,1],padding='SAME')
        biases=helper_variable_on_cpu('biases',[NUM_CLASSES],tf.constant_initializer(0.0))
        conv_classifier=tf.nn.bias_add(conv,biases,name=scope.name)
    
    logit=conv_classifier
    loss=cal_loss(conv_classifier,labels)
    
    return loss,logit
    

# train's network
def train(total_loss,global_step):
    total_sample=274
    num_batches_per_epoch = 274/1
    lr=INITIAL_LEARNING_RATE
    loss_average_op=helper_add_loss_summaries(total_loss=total_loss)
    
    # gradiens
    with tf.control_dependencies([loss_average_op]):
        opt=tf.train.AdamOptimizer(lr)
        grads=opt.compute_gradients(total_loss)
    
    apply_gradient_op=opt.apply_gradients(grads,global_step=global_step)
    
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op=variable_averages.appy(tf.trainable_variables())
    
    with tf.control_dependencies([apply_gradient_op,variable_averages_op]):
        train_op=tf.no_op(name='train')
        
    return train_op
    
    
    
'''------------------------------------------------------------------'''

def training(trainfilepath,valfilepath,batch_size,image_width,image_height,image_ch,max_steps):
    train_image_filenames,train_label_filenames=get_filename_list(trainfilepath)
    val_image_filenames,val_label_filenames=get_filename_list(valfilepath)
    startstep = 0
    with tf.Graph().as_default():
        # train_data_node和train_label_node
        #    这两者作为训练集的data和label.
        train_data_node=tf.placeholder(tf.float32,shape=[batch_size,image_height,image_width,image_ch])
        train_label_node=tf.placeholder(tf.float32,shape=[batch_size,image_height,image_width,1]) # 它是1个通道.

        #phase_train
        # phase_train作为conv*的输入.是一个True和false的
        phase_train=tf.placeholder(tf.float32,name="phase_train") # 为什么没有设置shape=[],因为它是Bool型变量
        global_step=tf.Variable(0,trainable=False) # 设置步长,不参与训练
        
        
        # 读出camVid
        train_images,train_labels=CamVidInput(train_image_filenames,train_label_filenames,batch_size)
        val_images,val_labels=CamVidInput(val_image_filenames,val_label_filenames,batch_size)
        
        
        
        # 建立encoder+decoder的网络图
        #     输入: data和y.
        #     返回损失和预测精度
        #     phase_train作用,是Bool型
        # train_data_node,train_label_node: 
        #     输入数据集及标签.
        loss,eval_prediction=inference(train_data_node,train_label_node,batch_size,phase_train)
        
        
        
        # 建立train的图
        train_op=train(loss,global_step)
        
        with tf.Session() as sess:
            init=tf.global_variables_initializer()
            sess.run(init)
            
            # 创建线程,并用coordinator()管理
            coord=tf.train.Coordinator()
            threads=tf.train.start_queue_runners(sess=sess,coord=coord)
            
            
            for step in range(startstep,startstep+max_steps):
                image_batch,label_batch=sess.run([train_images,train_labels])
                feed_dict={
                    train_data_node:image_batch,
                    train_label_node:label_batch,
                    phase_train:True
                }
                _,loss_value=sess.run([train_op,loss],feed_dict=feed_dict) # 第一个_是不关心的.
                if step%10==0:
                    print("setp:%d,loss=%.2f" %(step,loss_value))
                pred=sess.run(eval_prediction,feed_dict=feed_dict)
                print("pred:%s"%pred)
    
    coord.request_stop()
    coord.join(threads)
                
                
                
                
            

if __name__=='__main__':
    training(trainfilepath="/home/julyedu_433249/work/demo_playgroud/Tensorflow-SegNet/SegNet/CamVid/train.txt",
             valfilepath="/home/julyedu_433249/work/demo_playgroud/Tensorflow-SegNet/SegNet/CamVid/val.txt",
             batch_size=5,
             image_width=480,
             image_height=360,
             image_ch=3,
             max_steps=20000)
