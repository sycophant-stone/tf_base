import numpy as np
import os
import cv2
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
caffe.set_device(3)
caffe.set_mode_gpu()
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

def draw_bbox_on_image(nxmin,nymin,nxmax,nymax, img_filepath):
    img = cv2.imread(img_filepath)
    region_rect = cv2.rectangle(img, (int(nxmin),int(nymin)), (int(nxmax),int(nymax)), (0,255,0), 2)
    tmp_filepath = os.path.basename(img_filepath)
    rect_img_savepath = "example/res/"+os.path.splitext(tmp_filepath)[0]+"_rec.jpg"
    print(rect_img_savepath)
    cv2.imwrite(rect_img_savepath, region_rect)


if __name__ == '__main__':
    # load PASCAL VOC labels
    labelmap_file = '/ssd/hnren/tf/1sd/MobileNet-SSD/labelmap_voc.prototxt'
    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)
    
    
    model_def = 'example_5_patches300to128_INT_patches_inte/MobileNetSSD_deploy.prototxt'
    model_weights = 'snapshot/mobilenet_iter_105000.caffemodel'
    
    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)
    
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([127.5,127.5,127.5])) # mean pixel
    #transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    #transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    transformer.set_input_scale('data', 0.007843)  # the reference model has channels in BGR order instead of RGB
    
    # set net to batch size of 1
    image_resize = 128
    net.blobs['data'].reshape(1,3,image_resize,image_resize)
    
    #image = caffe.io.load_image('example/images/ch00005_20190303_ch00005_20190303105000.mp4.cut.mp4_001500_crop_0.jpg')
    #image = caffe.io.load_image('example/images/ch01014_20190322_ch01014_20190322084500.mp4.cut.mp4_000000_crop_4.jpg')
    jpgs = list_all_files("example/images", exts=["jpg"])
    for jpg_img in jpgs:
        image = caffe.io.load_image(jpg_img)
        image_basename = os.path.basename(jpg_img)
        #plt.imshow(image)
        
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        
        # Forward pass.
        detections = net.forward()['detection_out']
        
        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]
        print("%s's decttions res %s:"%(image_basename, detections))
        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
        
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]
        #print("top_conf", top_conf)
        #print("top_xmin", top_xmin)
        #print("top_ymin", top_ymin)
        #print("top_xmax", top_xmax)
        #print("top_ymax", top_ymax)
        draw_bbox_on_image(top_xmin*128, top_ymin*128, top_xmax*128, top_ymax*128, jpg_img)
        '''
        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        
        #plt.imshow(image)
        currentAxis = plt.gca()
        
        for i in xrange(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            display_txt = '%s: %.2f'%(label_name, score)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            color = colors[label]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
        
        #plt.imshow(image)
        '''
