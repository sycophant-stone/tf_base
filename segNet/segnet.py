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
import skimage
import skimage.io
# modules
#from Utils import _variable_with_weight_decay, _variable_on_cpu, _add_loss_summaries, _activation_summary, print_hist_summery, get_hist, per_class_acc, writeImage
#from Inputs import *

'''-------'''
# parma presetting
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480
IMAGE_DEPTH = 3
NUM_CLASSES=12

EVAL_BATCH_SIZE = 5
BATCH_SIZE = 5

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

INITIAL_LEARNING_RATE = 0.001 


NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 367
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 101
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1
TEST_ITER = NUM_EXAMPLES_PER_EPOCH_FOR_TEST / BATCH_SIZE

'''------------------------------------------------------------------'''
##input
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def per_class_acc(predictions, label_tensor):
    labels = label_tensor
    size = predictions.shape[0]
    num_class = predictions.shape[3]
    hist = np.zeros((num_class, num_class))
    for i in range(size):
      hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
    acc_total = np.diag(hist).sum() / hist.sum()
    print ('accuracy = %f'%np.nanmean(acc_total))
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print ('mean IU  = %f'%np.nanmean(iu))
    for ii in range(num_class):
        if float(hist.sum(1)[ii]) == 0:
          acc = 0.0
        else:
          acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
        print("    class # %d accuracy = %f "%(ii,acc))

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

def helper_generate_image_label_batch(images,labels,min_queue_examples,batch_size,shuffle=True):
    """
    有image和label以及batch_size.拿出一组 [btach_size,ih,iw,ch]的数据
    """ 
    print("min_queue_examples:",min_queue_examples)
    if shuffle:
        images_batch,label_batch=tf.train.shuffle_batch([images,labels],batch_size=batch_size,num_threads=1,capacity=min_queue_examples+3*batch_size,min_after_dequeue=min_queue_examples)
    else:
        images,label_batch=tf.train.batch([images,labels],batch_size=batch_size,num_threads=1,capacity=min_queue_examples+3*batch_size)
    print("label_batch shape:%d,%d,%d,%d"%(label_batch.shape[0],label_batch.shape[1],label_batch.shape[2],label_batch.shape[3]))
    return images_batch,label_batch


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
    # 丢掉了
    reshape_image=tf.cast(image,tf.float32)
    
    # 产生一组images和labels
    min_fraction_of_example_in_queue=0.4
    min_queue_examples=int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_example_in_queue)
    return helper_generate_image_label_batch(reshape_image,label,min_queue_examples,batch_size,shuffle=True)
    
def get_all_test_data(image_filenames,label_filenames):
    """根据image和label文件名路径,读取出对应文件,把它们整理成数据.
    args:
        image_filenames:
            测试集文件路径
        label_filenames:
            测试集的mask图,这个作为预测的真值.
    return:
        返回数组状态的images和labels.
    """
    images=[]
    labels=[]
    for im_filename,lb_filename in zip(image_filenames,label_filenames):
        im=np.array(skimage.io.imread(im_filename))
        im=im[np.newaxis] # 其实起到一个把数组变成矩阵的功能
        images.append(im)
        lb=skimage.io.imread(lb_filename)
        lb=lb[np.newaxis]
        lb=lb[...,np.newaxis]
        labels.append(lb)
    
    return images,labels


def writeImage(image, filename):
    """ store label data to colored image """
    Sky = [128,128,128]
    Building = [128,0,0]
    Pole = [192,192,128]
    Road_marking = [255,69,0]
    Road = [128,64,128]
    Pavement = [60,40,222]
    Tree = [128,128,0]
    SignSymbol = [192,128,128]
    Fence = [64,64,128]
    Car = [64,0,128]
    Pedestrian = [64,64,0]
    Bicyclist = [0,128,192]
    Unlabelled = [0,0,0]
    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array([Sky, Building, Pole, Road_marking, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
    for l in range(0,12):
        """
        image是预测值,它应该有一组区域构成,同一组区块用一个数字表示,比如0,会表示sky.比如靠近地面的地方会有一组4,表示road
        但是这个数字图是不能显示出来给人看的,那么我们可以针对每一组都一个它一个颜色,比如对于所有位置是1的区域,人为修改该区域的
        rbg为sky的Sky = [128,128,128],这样这种图可以被人眼看出来.
        处理时候还是会依照rgb来处理
        """
        r[image==l]=label_colours[l,0]
        g[image==l]=label_colours[l,1]
        b[image==l]=label_colours[l,2]
  
    # 保存图片
    rgb=np.zeros([image.shape[0],image.shape[1],3])
    rgb[:,:,0]=r/1.0
    rgb[:,:,1]=g/1.0
    rgb[:,:,2]=b/1.0
    im=Image.fromarray(np.uint8(rgb))
    im.save(filename)

    
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
    for l in losses+[total_loss]:
        tf.summary.scalar(l.op.name+' (raw)', l)
        tf.summary.scalar(l.op.name, loss_average.average(l))

    return loss_average_op
    
def batch_norm_layer(inputI,is_trainning,scope):
    return tf.cond(is_trainning,
                   lambda: tf.contrib.layers.batch_norm(inputI,is_training=True,center=False,updates_collections=None,scope=scope+"_bn"),
                   lambda: tf.contrib.layers.batch_norm(inputI,is_training=False,center=False,updates_collections=None,scope=scope+"_bn",reuse=True))

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
            conv_out=batch_norm_layer(bias,train_phase,scope.name)
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
        
        
def deconv_layer(inputI,f_shape,output_shape,stride=2,name=None):
    # 这个shape的格式: kernel_w,kernel_h,batch,chn
    sess_temp=tf.global_variables_initializer()
    strides=[1,stride,stride,1]
    with tf.variable_scope(name):
        weights=get_decode_filter(f_shape)
        deconv=tf.nn.conv2d_transpose(inputI,weights,output_shape,strides=strides,padding='SAME') # 这里strides区别于stride,strides是一组参数.
    
    return deconv

def msra_initializer(k1,d1):
    """
    k1: kernel size
    d1: filter number
    """
    stddev=math.sqrt(2. /(k1**2 * d1))
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
        """[[[特别注意]]]: 交叉熵是是正值,由于概率是小于1的,所以log的值是负值,所以需要一个负号来修正"""
        cross_entropy=  -tf.reduce_sum(tf.multiply(labels*tf.log(softmax+epsilon),head),axis=[1]) ### 注意交叉熵是正值,log操作前面要有负号来修证!!!!!!
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


def get_hist(predictions,labels):# labels不准确,这里传入的是batch_size
    num_cls=predictions.shape[3]
    batch_size=predictions.shape[0]
    hist=np.zeros([num_cls,num_cls])
    for i in range(batch_size):
        ylabel=labels[i].flatten()
        #print("ylabel",ylabel)
        yHat=predictions[i].argmax(2).flatten()
        #print("yHat",yHat)
        k=(ylabel>0)&(ylabel<num_cls) # sanity check , 找到同时满足0<?<num_cls的那些位的地址.
        #print("k",k)
        #print("len(k)",len(k))
        #print("labels.len",len(ylabel))
        #print("yHat.len",len(yHat))
        hist+=np.bincount(num_cls*ylabel[k].astype(int)+yHat[k],minlength=num_cls**2).reshape(num_cls,num_cls) # num_cls个类出现的次数.
    
    return hist
        
        

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
    upsample4=deconv_layer(pool4,[2,2,64,64],[batch_size,45,60,64],2,"up4")
    # decode4
    conv_decode4=conv_layer_with_bn(upsample4,[7,7,64,64],phase_train,False,"conv_decode4")
    
    # upsamping3
    upsample3=deconv_layer(conv_decode4,[2,2,64,64],[batch_size,90,120,64],2,"up3")
    # decode3
    conv_decode3=conv_layer_with_bn(upsample3,[7,7,64,64],phase_train,False,"conv_decode3")
    
    # upsamping2
    upsample2=deconv_layer(conv_decode3,[2,2,64,64],[batch_size,180,240,64],2,"up2")
    # decode2
    conv_decode2=conv_layer_with_bn(upsample2,[7,7,64,64],phase_train,False,"conv_decode2")
    
    # upsampling1
    upsample1=deconv_layer(conv_decode2,[2,2,64,64],[batch_size,360,480,64],2,"up1")
    # decode2
    conv_decode1=conv_layer_with_bn(upsample1,[7,7,64,64],phase_train,False,"conv_decode1")
    
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
    
    return loss,logit,norm1
    

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
    
    # 记录参与训练的变量的直方图
    for var in tf.trainable_variables():
        print("%s",var.op.name)
        tf.summary.histogram(var.op.name,var)
    
    # 记录梯度直方图
    for grad,var in grads:
        if grad is not None:
            stringname=var.op.name+"/gradients"
            print(stringname)
            tf.summary.histogram(var.op.name+"/gradients",grad)
        
    
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op=variable_averages.apply(tf.trainable_variables())
    
    with tf.control_dependencies([apply_gradient_op,variable_averages_op]):
        train_op=tf.no_op(name='train')
        
    return train_op

'''------------------------------------------------------------------'''

def test(testfilepath,batch_size,image_width,image_height,image_ch):
    """测试函数,会恢复training时候保存的ckpt模型文件
    testfilepath:
        含有测试集的照片列表文件
    batch_size:
        每次做几个文件的推断
    image_width,image_height,image_ch
        每张图的尺寸
    
    """
    # 读取测试集,分成image和label.
    test_image_filenames,test_label_filenames=get_filename_list(testfilepath)
    
    # 建立graph的测试节点
    test_data_node=tf.placeholder(tf.float32,shape=[batch_size,image_height,image_width,image_ch])
    test_label_node=tf.placeholder(tf.int64,shape=[batch_size,360,480,1]) # 只有单通道的mask图.
    phase_train=tf.placeholder(tf.bool,name="phase_train")
    
    # 给网络通入输入
    loss,logit,_=inference(test_data_node,test_label_node,batch_size,phase_train)
    
    pred=tf.argmax(logit,axis=3) # 取出预测最高的前三个结果
    
    # 从保存的网络中直接恢复到当前网络的权重中,而非恢复到影子变量中.
    variable_arverage=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variable_restore=variable_arverage.variables_to_restore()
    saver=tf.train.Saver(variable_restore)
   
    # 开始往网络中冠数据
    with tf.Session() as sess:
        # 加载网络图
        ####saver=tf.train.import_meta_graph('/home/julyedu_433249/work/log/segnet.model.ckpt-19999.meta') 
        saver.restore(sess,tf.train.latest_checkpoint('/home/julyedu_433249/work/log/'))
        tvs=[v for v in tf.trainable_variables()]
        for v in tvs:
            print(v.name)
            print(sess.run(v))

        # 加载网络
        #model_file=tf.train.latest_checkpoint("")
        #logdir="/home/julyedu_433249/work/log/checkpoint"
        #saver.restore(sess,logdir)
        
        # 加载图片和mask的label图
        # 区别于train时候利用tensorflow去读数据集,我们这里利用skimage包去读测试集
        test_images,test_labels=get_all_test_data(test_image_filenames,test_label_filenames)
        
        # 启动文件名队列填充过程
        tf.train.start_queue_runners(sess)
        
        # 统计正确率需要的hist
        hist= np.zeros((NUM_CLASSES,NUM_CLASSES))
        
        # 批量
        for image_batch,label_batch in zip(test_images,test_labels): # 注意理解和trian时候利用sess.run得到数据集的不同.
            feed_dict={
                test_data_node:image_batch,
                test_label_node:label_batch,
                phase_train:False
            }
            test_loss,test_logit,test_pred=sess.run([loss,logit,pred],feed_dict=feed_dict)
            
            # 保存预测区域
            writeImage(test_pred[0],"test_pred.png") # 保存第一个最大的预测
            hist+=get_hist(test_logit,label_batch)
        # 一组batch推断后,评估准确率
        acc_total=np.diag(hist).sum()/hist.sum()
        # 交并区域
        iu=np.diag(hist).sum()/(hist.sum(0)+hist.sum(1)-np.diag(hist).sum())
        print("acc:",acc_total)
        print("mean iu",np.mean(iu))
    
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
        phase_train=tf.placeholder(tf.bool,name="phase_train") # 为什么没有设置shape=[],因为它是Bool型变量,  phase_train是个bool型而非float32型.它描述当前是train还是test.true为train.
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
        loss,eval_prediction,norm1=inference(train_data_node,train_label_node,batch_size,phase_train)
        
        # 建立train的图
        train_op=train(loss,global_step)
        
        summary_op=tf.summary.merge_all() #把之前train的summary合到一起.
        
        # 记录average_loss,test_accuracy,mean_IU值,它们是标量.
        average_loss =tf.placeholder(tf.float32)
        accuracy=tf.placeholder(tf.float32)
        mean_iu=tf.placeholder(tf.float32)
        
       
        saver=tf.train.Saver(tf.global_variables())
        
        
        with tf.Session() as sess:
            init=tf.global_variables_initializer()
            sess.run(init)
            
            # 创建线程,并用coordinator()管理
            coord=tf.train.Coordinator()
            threads=tf.train.start_queue_runners(sess=sess,coord=coord)
            # tf.summary.scalar 会存储标量
            average_loss_smy=tf.summary.scalar("average_loss",average_loss)
            accuracy_smy=tf.summary.scalar("accuracy",accuracy)
            mean_iu_smy=tf.summary.scalar("mean_iu",mean_iu)
        
            # tf file writer
            logdir="/home/julyedu_433249/work/log"
            summary_filewriter=tf.summary.FileWriter(logdir,sess.graph)
            
            
            for step in range(startstep,startstep+max_steps):
                image_batch,label_batch=sess.run([train_images,train_labels])
                feed_dict={
                    train_data_node:image_batch,
                    train_label_node:label_batch,
                    phase_train:True
                }
                #print("train_data_node shape",train_data_node.shape())
                #print("image_batch shape",image_batch.shape())
                _,loss_value=sess.run([train_op,loss],feed_dict=feed_dict) # 第一个_是不关心的.
                if step==0:
                    print("setp:%d,loss=%.2f" %(step,loss_value))
                    
                if step%10==0:
                    print("setp:%d,loss=%.2f" %(step,loss_value))
                    pred=sess.run(eval_prediction,feed_dict=feed_dict)
                    per_class_acc(pred,label_batch)
                    """For debug
                    norm1_val=sess.run(norm1,feed_dict=feed_dict)
                    print("norm1_val",norm1_val)
                    """
                if step%100==0:#每100次做一下验证集.计算误差,精度等.
                    #print("pred:%s"%pred)
                    print("start validation")
                    total_val_loss=0.0
                    hist=np.zeros([NUM_CLASSES,NUM_CLASSES])
                    for val_step in range(int(TEST_ITER)):
                        val_image_batch,val_label_batch=sess.run([val_images,val_labels])
                        val_loss,val_pred=sess.run([loss,eval_prediction],feed_dict={
                            train_data_node:val_image_batch,
                            train_label_node:val_label_batch,
                            phase_train:True
                        })
                        #print("val_pred shape:%d,%d,%d,%d"%(val_pred.shape[0],val_pred.shape[1],val_pred.shape[2],val_pred.shape[3]))
                        #print("val_image_batch shape:%d,%d,%d,%d"%(val_image_batch.shape[0],val_image_batch.shape[1],val_image_batch.shape[2],val_image_batch.shape[3]))
                        total_val_loss+=val_loss
                        hist+=get_hist(val_pred,val_label_batch) #val_image_batch)  注意此处是val_label_batch,而非image.这个花了我们好长时间.
                    print("hist:",hist)
                    print("np.diag(hist)",np.diag(hist))
                    print("hist.sum(0)",hist.sum(0))
                    print("hist.sum(1)",hist.sum(1))
                    acc_total=np.diag(hist).sum()/hist.sum()
                    iu=np.diag(hist)/(hist.sum(0)+hist.sum(1)-np.diag(hist))
                    average_loss_smy_str=sess.run(average_loss_smy,feed_dict={
                        average_loss:total_val_loss/TEST_ITER
                    })
                    accuracy_smy_str=sess.run(accuracy_smy,feed_dict={
                        accuracy:acc_total
                    })
                    mean_iu_smy_str=sess.run(mean_iu_smy,feed_dict={
                        mean_iu:np.nanmean(iu) # 注意这里有np.nanmean对iu的均值.不包含nan的数值.
                    })
                    summary_str=sess.run(summary_op,feed_dict=feed_dict)
                    summary_filewriter.add_summary(summary_str,step)
                    summary_filewriter.add_summary(average_loss_smy_str,step)
                    summary_filewriter.add_summary(accuracy_smy_str,step)
                    summary_filewriter.add_summary(mean_iu_smy_str,step)
                
                if step%1000==0 or (step+1)==max_steps:
                    # 保存模型
                    checkpoint_path=os.path.join(logdir,"segnet.model.ckpt")
                    saver.save(sess,checkpoint_path,global_step=step)
                    
                    
                    
    
    coord.request_stop()
    coord.join(threads)
                
                
                
                
            

if __name__=='__main__':
    
    if len(sys.argv) < 2:
        print ("NO action specified.")
        sys.exit()

    if sys.argv[1].startswith('--'):
        option = sys.argv[1][2:]
        if option == 'version':
            print ("version 1.2 ")
        elif option == 'help':
            print ("This program prints files to the standard output.\
                 Any number of files can be specified.\
                 Options include:\
                 --version : Prints the version number\
                 --train: traing segnet\
                 --test: test segnet\
                 --help     : Display this help")
            
        elif option == 'train':
            print("start training")
            training(trainfilepath="/home/julyedu_433249/work/tf_base/segNet/SegNet/CamVid/train.txt",
             valfilepath="/home/julyedu_433249/work/tf_base/segNet/SegNet/CamVid/val.txt",
             batch_size=5,
             image_width=480,
             image_height=360,
             image_ch=3,
             max_steps=20000)
        elif option == 'test':
            print("start testing")
            test(testfilepath="/home/julyedu_433249/work/tf_base/segNet/SegNet/CamVid/test.txt",
             batch_size=1,
             image_width=480,
             image_height=360,
             image_ch=3)

        else:
            print("Unknow option.")
    
    

