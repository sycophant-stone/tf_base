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
    print("prebbox:(%d,%d)to(%d,%d), area:%d" %(prediction_bbox[0],prediction_bbox[1],prediction_bbox[2],prediction_bbox[3],prebboxarea))
    print("gtbbox :(%d,%d)to(%d,%d), area:%d" %(gt_bbox[0],gt_bbox[1],gt_bbox[2],gt_bbox[3],gtbboxarea))

    iou = (xmin-xmax)*(ymin-ymax)
    iou = iou/(prebboxarea+gtbboxarea-iou)
    if iou < 0:
        raise Exception("iou is Negtive")
        iou = 0
    print("iou:",iou)
    return iou    
    
def find_col_maxvalue(prediction_bbox,gt_bbox):
    xmax_ = np.maximum(prediction_bbox[:,0],gt_bbox[:,0])
    print("xmax_", xmax_)
    print("xmax_'s shape", xmax_.shape)

    ymax_ = np.maximum(prediction_bbox[:,1],gt_bbox[:,1])
    print("ymax_", ymax_)
    print("ymax_'s shape", ymax_.shape)

    xmin_ = np.minimum(prediction_bbox[:,2],gt_bbox[:,2])
    print("xmin_", xmin_)
    print("xmin_'s shape", xmin_.shape)

    ymin_ = np.minimum(prediction_bbox[:,3],gt_bbox[:,3])
    print("ymin_", ymin_)
    print("ymin_'s shape", ymin_.shape)

    xcond_ = np.less(xmax_, xmin_)
    print("xcond_", xcond_)
    xleft_ = np.where(xcond_, xmax_, np.zeros(xcond_.shape))
    print("xleft_", xleft_)

    ycond_ = np.less(ymax_, ymin_)
    print("ycond_", ycond_)
    yleft_ = np.where(ycond_, ymax_, np.zeros(ycond_.shape))
    print("yleft_", yleft_)



    return 
    
def test_vectorize_calc_iou():
    print("enter test_vectorize_calc_iou")
    pred = np.array([[287.37198 , 118.55582 , 291.1561  , 190.13669 ],
       [580.10376 , 142.63705 , 885.83923 , 146.14606 ],
       [291.43027 , 123.61223 , 347.4736  , 130.00053 ],
       [300.77795 , -10.516249, 313.15857 ,  58.1938  ],
       [100.60458 , 146.97646 , 173.40833 , 161.21843 ],
       [86.82617 , 179.19148 , 134.84885 , 193.90288 ]], dtype="float32")
    #gt = np.array([[262.94,239,62.04001,70.40001]], dtype="float32")
    gt = np.array([[62.04001,70.40001,262.94,239]], dtype="float32")
    print("pred shape:", pred.shape)
    print("pred value:", pred)
    print("gt shape:", gt.shape)
    print("gt value:", gt)
    #iou = calc_iou(pred,gt)
    #print("iou:",iou)
    find_col_maxvalue(pred, gt)
    
if __name__ == '__main__':
    test_vectorize_calc_iou()
