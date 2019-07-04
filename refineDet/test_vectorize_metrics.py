from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow as wrap
import sys
import os
import numpy as np

def calc_iou(prediction_bbox, gt_bbox):
    """calc iou between prediction's bbox and groud truth's bbox
       @param:
       prediction_bbox:
           prediction boundingbox, [400, 4]
       gt_bbox:
           groudtruth boundingbox, [60, 4]
    """
    iou = 0
    #np.maxmium(prediction_bbox[:,0],gt_bbox[:,0])
    if prediction_bbox[0]>gt_bbox[0]:
        xmax = gt_bbox[0]
    else:
        xmax = prediction_bbox[0]
    if prediction_bbox[2]>gt_bbox[2]:
        xmin = gt_bbox[2]
    else:
        xmin = prediction_bbox[2]
    if prediction_bbox[1]<gt_bbox[1]:
        xmax = gt_bbox[1]
    else:
        xmax = prediction_bbox[1]
    if prediction_bbox[3]>gt_bbox[3]:
        xmin = gt_bbox[3]
    else:
        xmin = prediction_bbox[3]
    
    if xmax>xmin:
        return 0
    if ymax>ymin:
        return 0
    prebboxarea = (prediction_bbox[2] - prediction_bbox[0])*(prediction_bbox[3] - prediction_bbox[1])
    gtbboxarea = (gt_bbox[2] - gt_bbox[0])*(gt_bbox[3] - gt_bbox[1])
    #print("prebbox:(%d,%d)to(%d,%d), area:%d" %(prediction_bbox[0],prediction_bbox[1],prediction_bbox[2],prediction_bbox[3],prebboxarea))
    #print("gtbbox :(%d,%d)to(%d,%d), area:%d" %(gt_bbox[0],gt_bbox[1],gt_bbox[2],gt_bbox[3],gtbboxarea))

    iou = (xmin-xmax)*(ymin-ymax)
    iou = iou/(prebboxarea+gtbboxarea-iou)
    if iou < 0:
        raise Exception("iou is Negtive")
        iou = 0
    #print("iou:",iou)
    return iou    
def tf_calc_iou_vectorized(prediction_bbox,gt_bbox):
    xmax_ = tf.maximum(prediction_bbox[:,0],gt_bbox[:,0])
    #print("xmax_", xmax_)
    #print("xmax_'s shape", xmax_.shape)

    ymax_ = tf.maximum(prediction_bbox[:,1],gt_bbox[:,1])
    #print("ymax_", ymax_)
    #print("ymax_'s shape", ymax_.shape)

    xmin_ = tf.minimum(prediction_bbox[:,2],gt_bbox[:,2])
    #print("xmin_", xmin_)
    #print("xmin_'s shape", xmin_.shape)

    ymin_ = tf.minimum(prediction_bbox[:,3],gt_bbox[:,3])
    #print("ymin_", ymin_)
    #print("ymin_'s shape", ymin_.shape)

    xcond_ = tf.less(xmax_, xmin_)
    #print("xcond_", xcond_)
    xleft_ = tf.where(xcond_, xmax_, tf.zeros(xcond_.shape))
    xdelta_ = tf.where(xleft_, tf.subtract(xmin_, xmax_), tf.zeros(xcond_.shape))
    #print("xleft_", xleft_)
    #print("xdelta_", xdelta_)

    ycond_ = tf.less(ymax_, ymin_)
    #print("ycond_", ycond_)
    yleft_ = tf.where(ycond_, ymax_, tf.zeros(ycond_.shape))
    ydelta_ = tf.where(yleft_, tf.subtract(ymin_, ymax_), tf.zeros(ycond_.shape))
    #print("yleft_", yleft_)
    #print("ydelta_", ydelta_)

    prediction_area = tf.multiply(tf.subtract(prediction_bbox[:,2], prediction_bbox[:,0]), tf.subtract(prediction_bbox[:,3], prediction_bbox[:,1]))
    gt_area = tf.multiply(tf.subtract(gt_bbox[:,2], gt_bbox[:,0]), tf.subtract(gt_bbox[:,3], gt_bbox[:,1]))
    iou = tf.multiply(xdelta_, ydelta_)
    iou = iou/(prediction_area + gt_area - iou)
    #print("prediction_area:", prediction_area)
    #print("gt_area:", gt_area)
    #print("iou:",iou)
    return iou

    
def calc_iou_vectorized(prediction_bbox,gt_bbox):
    if len(gt_bbox.shape) > 2:
        print("gtbbox shape doesn't match")
        gt_bbox = np.squeeze(gt_bbox)
    if gt_bbox.shape[1] != 4:
        print("gtbbox's colum doesn't match")
        colm=[0,1,2,3]
        gt_bbox = gt_bbox[:,colm]
    checksum=0
    for itm in range(gt_bbox.shape[0]):
        if gt_bbox[itm,0] !=-1:
            checksum=checksum+1
    if checksum>1:
        raise Exception("gt boxes have more than one sample")
    gt_bbox = gt_bbox[0,:]
    print("prediction shape: ", prediction_bbox.shape)
    print("gt_bbox shape: ", gt_bbox.shape)
    print("gt ",gt_bbox)
    xmax_ = np.maximum(prediction_bbox[:,0],gt_bbox[0])
    #print("xmax_", xmax_)
    #print("xmax_'s shape", xmax_.shape)

    ymax_ = np.maximum(prediction_bbox[:,1],gt_bbox[1])
    #print("ymax_", ymax_)
    #print("ymax_'s shape", ymax_.shape)

    xmin_ = np.minimum(prediction_bbox[:,2],gt_bbox[2])
    #print("xmin_", xmin_)
    #print("xmin_'s shape", xmin_.shape)

    ymin_ = np.minimum(prediction_bbox[:,3],gt_bbox[3])
    #print("ymin_", ymin_)
    #print("ymin_'s shape", ymin_.shape)

    xcond_ = np.less(xmax_, xmin_)
    #print("xcond_", xcond_)
    xleft_ = np.where(xcond_, xmax_, np.zeros(xcond_.shape))
    xdelta_ = np.where(xleft_, xmin_-xmax_, np.zeros(xcond_.shape))
    #print("xleft_", xleft_)
    #print("xdelta_", xdelta_)

    ycond_ = np.less(ymax_, ymin_)
    #print("ycond_", ycond_)
    yleft_ = np.where(ycond_, ymax_, np.zeros(ycond_.shape))
    ydelta_ = np.where(yleft_, ymin_-ymax_, np.zeros(ycond_.shape))
    #print("yleft_", yleft_)
    #print("ydelta_", ydelta_)

    prediction_area = (prediction_bbox[:,2] - prediction_bbox[:,0])*(prediction_bbox[:,3] - prediction_bbox[:,1])
    gt_area = (gt_bbox[2] - gt_bbox[0])*(gt_bbox[3] - gt_bbox[1])
    iou = xdelta_* ydelta_
    iou = iou/(prediction_area + gt_area - iou)
    #print("prediction_area:", prediction_area)
    #print("gt_area:", gt_area)
    #print("iou:",iou)
    return iou
    
def test_vectorize_calc_iou():
    #print("enter test_vectorize_calc_iou")
    pred = np.array([[287.37198 , 118.55582 , 291.1561  , 190.13669 ],
       [580.10376 , 142.63705 , 885.83923 , 146.14606 ],
       [291.43027 , 123.61223 , 347.4736  , 130.00053 ],
       [300.77795 , -10.516249, 313.15857 ,  58.1938  ],
       [100.60458 , 146.97646 , 173.40833 , 161.21843 ],
       [86.82617 , 179.19148 , 134.84885 , 193.90288 ]], dtype="float32")
    #gt = np.array([[262.94,239,62.04001,70.40001]], dtype="float32")
    gt = np.array([[62.04001,70.40001,262.94,239]], dtype="float32")
                   #[164.48962, 144.80002,    64.629074,  75.240005]], dtype="float32")
    #print("pred shape:", pred.shape)
    print("pred value:", pred)
    #print("gt shape:", gt.shape)
    print("gt value:", gt)
    #iou = calc_iou(pred,gt)
    ##print("iou:",iou)
    iou = calc_iou_vectorized(pred, gt)
    print("iou:",iou)
    
if __name__ == '__main__':
    test_vectorize_calc_iou()
