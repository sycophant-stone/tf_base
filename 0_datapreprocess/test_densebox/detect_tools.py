import os
import sys
sys.path.append('.')
sys.path.append('/ssd/xulifeng/caffe/python')
import caffe
import cv2
import numpy as np

'''
Function:
	calculate multi-scale and limit the maxinum side to 1200 
Input: 
	img: original image
Output:
	scales  : Multi-scale
'''
def calculate_scales(img, factor, min_size):
    caffe_img = img.copy()
    h,w,ch = caffe_img.shape
    pr_scale = min(1.0, 1600.0/w)  # 1.0
    w = int(w*pr_scale)
    h = int(h*pr_scale)

    #multi-scale
    scales = []
    factor_count = 0
    minl = min(h,w)
    while minl >= min_size:
        scales.append(pr_scale*pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    return scales

    
def detect_multiscale(net, img, threshold, scale_factor, stride, label_width, label_height):
    origin_h,origin_w,ch = img.shape
    crop_width = int(label_width * stride)
    crop_height = int(label_height * stride)
    
    scales = calculate_scales(img, scale_factor, min(crop_width, crop_height))
    crops = []

    for scale in scales:
        hs = int(origin_h*scale)
        ws = int(origin_w*scale)
        
        if hs < crop_height or ws < crop_width:
            continue
        
        scale_img = cv2.resize(img,(ws,hs))
        caffe_img = scale_img.copy() / 256.0
        caffe_img = caffe_img.transpose((2,0,1))
        net.blobs['data'].reshape(1,3,hs,ws)
        net.blobs['data'].data[...]=caffe_img
        out_ = net.forward()
        map = out_['out'][0][1]
        map_h = map.shape[0]
        map_w = map.shape[1]
        
        selected = []
        for my in range(map_h):
            for mx in range(map_w):
                if (map[my, mx] >= threshold):
                    selected.append((my, mx))
                    xmin = max(0, mx - (label_width / 2))
                    xmax = xmin + label_width
                    ymin = max(0, my - (label_height / 2))
                    ymax = ymin + label_height
                    # clear the scores in the cropped region
                    map[max(0,ymin):min(map_h,ymax), max(0,xmin):min(map_w,xmax)] = 0
        
        for (my, mx) in selected:
            xmin = min(max(0, (mx - (label_width / 2)) * stride), ws - crop_width)
            ymin = min(max(0, (my - (label_height / 2)) * stride), hs - crop_height)
            crops.append(scale_img[ymin:ymin+crop_height, xmin:xmin+crop_width])

        if len(crops) >= 20:
            break
            
    return crops

def load_net(net_deploy, net_caffemodel, device_id = 0):
    net = caffe.Net(net_deploy,net_caffemodel,caffe.TEST)
    caffe.set_device(device_id)
    caffe.set_mode_gpu()
    return net

