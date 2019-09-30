import numpy as np
import os
import cv2
import time
import decimal
import argparse
import shell
#from log import *
import logging
from pascal_voc import PascalVocAnn
from eval_fppi import *
from shell import run_system_command
# save with datetime info
from datetime import datetime
from cputime import calc_mean


'''USAGE:
 python face_ssd_iterations.py --conf 0.6 --deploy m.deploy.prototxt --weights snapshot/M_iter_100000.caffemodel --draw True --loglevel 30 --iouthd 0.4 > log_c5i4 2>&1
'''

import caffe
#caffe.set_device(3)
#caffe.set_mode_gpu()
caffe.set_mode_cpu()
from google.protobuf import text_format
from caffe.proto import caffe_pb2

# list all image files
def list_all_files(dir_name, exts = ['jpg', 'bmp', 'png', 'xml', 'log']):
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
    #logging.info( "rec1, xmin,ymin:(%d,%d) xmax, ymax:(%d,%d)"%(rec1[0], rec1[1],rec1[2],rec1[3]))
    #logging.info( "rec2, xmin,ymin:(%d,%d) xmax, ymax:(%d,%d)"%(rec2[0], rec2[1],rec2[2],rec2[3]))

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
        logging.info("txmin:%d, txmax:%d, tymin:%d, tymax:%d"%(txmin, txmax, tymin, tymax))
        return 0
    else:
        intersect = (txmax- txmin) * (tymax- tymin)
        return (float(intersect) / (sum_area - intersect))*1.0

def find_closest_point(gtbbox_list,predbbox_list):
    '''find closest point
    input:
        gtbbox_list:   [gtxmin, gtymin, gtxmax, gtymax]
        predbbox_list: [xmin, ymin, xmax, ymax]
    output:
        x,y
    '''
    gtxmin, gtymin, gtxmax, gtymax = gtbbox_list
    center_x = (gtxmax+gtxmin)/2
    center_y = (gtxmax+gtxmin)/2

    pdxmin, pdymin, pdxmax, pdymax = predbbox_list

    outx= pdxmin if abs(pdxmin-center_x) < abs(pdxmax-center_x) else pdxmax
    outy= pdymin if abs(pdymin-center_y) < abs(pdymax-center_y) else pdymax
    
    return outx, outy

def put_iou_on_img(bboxes, img, predbbox_list):
    '''gen iou for every predbox, then put iou on its corresponding points
       it will ignore the iou below 0.1, to reduce meaningless text.
    input:
        bboxes:  gt boxes
        img:     image
        predbbox_list:   [xmin, ymin, xmax, ymax]
    return:
        img:      with iou value on the img's corresponding points
    '''
    for b in bboxes:
        gtxmin, gtymin, gtxmax, gtymax=b[1:5]
        iou = compute_iou([gtxmin, gtymin, gtxmax, gtymax], predbbox_list)
        if iou < 0.1:
            continue
        x,y=find_closest_point([gtxmin, gtymin, gtxmax, gtymax],predbbox_list)
        img = cv2.putText(img, "%.2f"%(iou), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)

    return img


'''
usage:

imgdir:
    JPEGImages:    for test set's images.
    Annotations:   for test set's annotations.
    
'''
if __name__ == '__main__':
    # load PASCAL VOC labels
    parse = argparse.ArgumentParser()
    parse.add_argument('--imgdir', type=str, help='test images dir, ex: /ssd/hnren/tf/1sd/1_lf/libfacedetection-master/all_head_in72/', required=True)
    parse.add_argument('--conf', type=str, help='threshold for classfication --conf 0.6', required=True)
    parse.add_argument('--deploy', type=str, help='m.deploy.prototxt', required=True)
    parse.add_argument('--weights', type=str, help='snapshot/mT**.caffemodel', required=True)
    parse.add_argument('--draw', type=str, help='whether output the prediction bbox GUI pictures')
    parse.add_argument('--iouthd', type=float, help='iou thred', default='0.5')
    parse.add_argument('--loglevel', type=int, help='loglevel, 30: WARNING and above; 20:INFO LOG', default=logging.WARNING)
    _args = parse.parse_args()
    input_imgdir   =  _args.imgdir
    input_conf     =  _args.conf
    input_deploy   =  _args.deploy
    input_weights  =  _args.weights
    input_draw     =  _args.draw
    input_iouthd   =  _args.iouthd
    log_level      =  _args.loglevel;
    
    iterations_num = os.path.splitext(os.path.basename(input_weights))[0].split('_')[2]
    save_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    which_set = os.path.splitext(os.path.basename(__file__))[0].split('_')[2]
    log_save_filename = 'snapshot/m_%s_iter%s_conf%s_iou%s_%s.log'%(which_set, iterations_num, input_conf, input_iouthd, save_timestamp)
    logging.basicConfig(filename=log_save_filename, format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = log_level, filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')

    labelmap_file = '/ssd/hnren/tf/1sd/MobileNet-SSD/labelmap_voc.prototxt'
    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)
    recall_rate_list=[]
    tp_list =[]
    fp_list =[]
    fn_list =[]
    if not os.path.exists(input_deploy):
        raise Exception("%s donesn't exist"%(input_deploy))
    if not os.path.exists(input_weights):
        raise Exception("%s donesn't exist"%(input_weights))
 
    model_def = input_deploy
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

    imglist_prefix=input_imgdir #"/ssd/hnren/tf/1sd/1_lf/libfacedetection-master/all_head_in72/"
    
    jpgs = list_all_files(imglist_prefix + "/JPEGImages/", exts=["jpg"])
    
    label_dict={}  
    dt_dict={}  
    gtdict={}
    pred_dict={}
    save_dstpath = imglist_prefix + "/facessdITERATION_conf%s_iou%s_%s/"%(input_conf, input_iouthd, save_timestamp)
    if not os.path.exists(save_dstpath):
        os.mkdir(save_dstpath)

    for jpg_img in jpgs:
        sample_basename = os.path.basename(jpg_img)
        xml_temp = os.path.splitext(sample_basename)[0]
        xml_sample_path =imglist_prefix + "/Annotations/"+xml_temp+".xml"
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
        logging.warning('Running time: {} Seconds'.format(end-start))
        # Parse the outputs.
        logging.info( detections)
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]
        if input_draw:
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= float(input_conf) ]
            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_labels = get_labelname(labelmap, top_label_indices)
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]
            logging.info( "input conf:%s, len top_indeces: %d"%(input_conf, len(top_indices)))
            img = cv2.imread(jpg_img)
            if len(top_indices)>=1:
                for xmin,ymin,xmax,ymax in zip(top_xmin, top_ymin, top_xmax, top_ymax):
                    img = draw_bbox_on_image(xmin*72, ymin*72, xmax*72, ymax*72, img)
                    img = put_iou_on_img(bboxes, img, [xmin*72, ymin*72, xmax*72, ymax*72])
                tmp_filepath = os.path.basename(jpg_img)
                rect_img_savepath = save_dstpath+os.path.splitext(tmp_filepath)[0]+"_rec.jpg"
                logging.info( "save to :%s"%(rect_img_savepath))
            else:
                tmp_filepath = os.path.basename(jpg_img)
                rect_img_savepath = save_dstpath+os.path.splitext(tmp_filepath)[0]+"_rec.jpg"
                logging.info( "save to :%s"%(rect_img_savepath))
             
            for b in bboxes:
                gtxmin,gtymin,gtxmax,gtymax=b[1:5]
                img = cv2.rectangle(img, (int(gtxmin),int(gtymin)), (int(gtxmax), int(gtymax)), (0,0,255), 1)
 
            cv2.imwrite(rect_img_savepath, img)
        #logging.warning("%s's decttions res %s:"%(image_basename, detections))
        # Get detections with confidence higher than 0.6.
        gtlist=[]
        for b in bboxes:
            box_item = b[1:5]
            box_item.append(0)
            gtlist.append(box_item)
        #logging.warning("gtlist", gtlist)
        gtdict[sample_basename]=gtlist
        
        pdlist=[]
        for d in detections[0,0,:]:
            if d[2] >= float(input_conf):
                pd=[d[3]*72, d[4]*72, d[5]*72, d[6]*72, d[2]]
                pdlist.append(pd)
        '''
        d=detections[0,0,0,:]
        pd=[d[3]*72, d[4]*72, d[5]*72, d[6]*72, d[2]]
        pdlist.append(pd)
        '''
        pred_dict[sample_basename]=pdlist        
        
    label_dict["head"]=gtdict
    dt_dict["head"]=pred_dict
    logging.info( "label_dict:%s"%(label_dict))
    logging.info( "dt_dict:%s"%(dt_dict))
    #match = detection_result_match(label_dict, dt_dict, 0.5)
    match = detection_result_match(label_dict, dt_dict, float(_args.iouthd))#0.45)
    logging.info( match)
    testfppi_val=[0.1, 0.25, 0.3]
    fppi_res = calculate_fppi(match, testfppi_val)
    logging.warning(fppi_res)
    logging.warning(fppi_res['head']['recall'][0])
    logging.warning(fppi_res['head']['recall'][1])
    logging.warning(fppi_res['head']['recall'][2])
    
    # output post process
    print('output log and model post-process ...')
    cmd = 'cp %s %s/'%(log_save_filename, save_dstpath)
    run_system_command(cmd)
    cmd = 'cp %s %s/'%(input_deploy, save_dstpath)
    run_system_command(cmd)
    cmd = 'cp %s %s/'%(input_weights, save_dstpath)
    run_system_command(cmd)
    print('save to %s...'%(save_dstpath))

    result_log = list_all_files(save_dstpath, exts=['log'])
    calc_mean(result_log[0])
    print('... done')

