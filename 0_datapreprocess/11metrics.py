import numpy as np
import os
import cv2
import time
import decimal
import argparse

from pascal_voc import PascalVocAnn
from eval_fppi import *


#import matplotlib.pyplot as plt

#plt.rcParams['figure.figsize'] = (10, 10)
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
'''
caffe_root = '../caffe'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')
'''
import caffe
#caffe.set_device(3)
#caffe.set_mode_gpu()
caffe.set_mode_cpu()
from google.protobuf import text_format
from caffe.proto import caffe_pb2

# list all image files
def list_all_files(dir_name, exts = ['jpg', 'bmp', 'png', 'xml']):
    result = []
    for dir, sub_dirs, file_names in os.walk(dir_name):
        for file_name in file_names:
            if any(file_name.endswith(ext) for ext in exts):
                result.append(os.path.join(dir, file_name))
    return result

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

def draw_bbox_on_image(nxmin,nymin,nxmax,nymax,img ):
    #print(img_filepath)
    region_rect = cv2.rectangle(img, (int(nxmin),int(nymin)), (int(nxmax),int(nymax)), (0,255,0), 1)
    return region_rect

def compute_iou(rec1, rec2):
    """
    computing IoU
                 xmin, ymin, xmax, ymax
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    rec1[0] = float(rec1[0])
    rec1[1] = float(rec1[1])
    rec1[2] = float(rec1[2])
    rec1[3] = float(rec1[3])
    print("rec1, xmin,ymin:(%d,%d) xmax, ymax:(%d,%d)"%(rec1[0], rec1[1],rec1[2],rec1[3]))
    print("rec2, xmin,ymin:(%d,%d) xmax, ymax:(%d,%d)"%(rec2[0], rec2[1],rec2[2],rec2[3]))

    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    txmin = max(rec1[0], rec2[0])
    tymin = max(rec1[1], rec2[1])
    txmax = min(rec1[2], rec2[2])
    tymax = min(rec1[3], rec2[3])
 
    # judge if there is an intersect
    if txmin >= txmax or tymin >= tymax:
        print("txmin:%d, txmax:%d, tymin:%d, tymax:%d"%(txmin, txmax, tymin, tymax))
        return 0
    else:
        intersect = (txmax- txmin) * (tymax- tymin)
        return (float(intersect) / (sum_area - intersect))*1.0
'''
usage:
    
'''
if __name__ == '__main__':
    # load PASCAL VOC labels
    parse = argparse.ArgumentParser()
    parse.add_argument('--conf', type=str)
    parse.add_argument('--deff', type=str)
    parse.add_argument('--weights', type=str)
    _args = parse.parse_args()
    input_conf = _args.conf
    input_def = _args.deff
    input_weights = _args.weights;

    labelmap_file = '/ssd/hnren/tf/1sd/MobileNet-SSD/labelmap_voc.prototxt'
    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)
    recall_rate_list=[]
    tp_list =[]
    fp_list =[]
    fn_list =[]
    if not os.path.exists(input_def):
        raise Exception("%s donesn't exist"%(input_def))
    if not os.path.exists(input_weights):
        raise Exception("%s donesn't exist"%(input_weights))
 
    model_def = input_def
    model_weights = input_weights
    
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)
    
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([127.5,127.5,127.5])) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    transformer.set_input_scale('data', 0.007843)  # the reference model has channels in BGR order instead of RGB
    
    # set net to batch size of 1
    image_resize = 72
    net.blobs['data'].reshape(1,3,image_resize,image_resize)
    
    jpgs = list_all_files("/ssd/hnren/tf/1sd/1_lf/libfacedetection-master/example/all_head_in72", exts=["jpg"])
    
    label_dict={}  
    dt_dict={}  
    gtdict={}
    pred_dict={}

    for jpg_img in jpgs:
        sample_basename = os.path.basename(jpg_img)
        xml_temp = os.path.splitext(sample_basename)[0]
        xml_sample_path = "/ssd/hnren/tf/1sd/caffe/data/5_head/HeadVocFormat/FID_DID_HEAD_CLEAN_0_patches_int/Annotations/"+xml_temp+".xml"
        pascal_voc_ann = PascalVocAnn(xml=xml_sample_path)
        bboxes = pascal_voc_ann.get_boxes()

        image = caffe.io.load_image(jpg_img)
        image_basename = os.path.basename(jpg_img)
        
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        start =time.time()
        # Forward pass.
        detections = net.forward()['detection_out']
        end =time.time()
        print('Running time: {} Seconds'.format(end-start))
        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]
        if DRAW:
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_labels = get_labelname(labelmap, top_label_indices)
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]
 
        #print("%s's decttions res %s:"%(image_basename, detections))
        # Get detections with confidence higher than 0.6.
        gtlist=[]
        for b in bboxes:
            box_item = b[1:5]
            box_item.append(0)
            gtlist.append(box_item)
        #print("gtlist", gtlist)
        gtdict[sample_basename]=gtlist
        
        pdlist=[]
        for d in detections[0,0,:]:
            #print("d:",d)
            pd=[d[3]*72, d[4]*72, d[5]*72, d[6]*72, d[2]]
            #print("pd",pd)
            pdlist.append(pd)
        '''
        d=detections[0,0,0,:]
        pd=[d[3]*72, d[4]*72, d[5]*72, d[6]*72, d[2]]
        pdlist.append(pd)
        '''
        pred_dict[sample_basename]=pdlist        
        
    label_dict["head"]=gtdict
    dt_dict["head"]=pred_dict
    #print("label_dict", label_dict)
    #print("dt_dict", dt_dict)
    match = detection_result_match(label_dict, dt_dict, 0.5)
    print(match)
    testfppi_val=[0.1, 0.25, 0.3]
    fppi_res = calculate_fppi(match, testfppi_val)
    print(fppi_res)

