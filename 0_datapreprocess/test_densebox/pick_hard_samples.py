import os, sys
sys.path.append('.')
sys.path.append('/ssd/xulifeng/caffe/python')
import caffe
import cv2, shutil
import numpy as np
import numpy.random as npr
import math


#################################################################

def mkdir(dr):
    if not os.path.exists(dr):
        os.mkdir(dr)

def net_predict(net, img):
    caffe_img = img.copy() / 256.0
    caffe_img = caffe_img.transpose((2, 0, 1))
    net.blobs['data'].data[...] = caffe_img
    out_ = net.forward()
    map = out_['out'][0][1]
    map_h = map.shape[0]
    map_w = map.shape[1]        
    score = 0
    for my in range(map_h):
        for mx in range(map_w):
            score = max(score, map[my, mx])
    return score

def collect_hard_negatives():
    out_dir = "/ssd/xulifeng/workspace/hd_densebox/tmp"
    neg_img_dir = "/ssd/xulifeng/workspace/hd_densebox/train_data/negatives"
    mkdir(out_dir)
    fout = open(os.path.join(out_dir, "picked.txt"), 'w')
    
    deploy = '/ssd/xulifeng/workspace/hd_densebox/train/net1_deploy.prototxt'
    caffemodel = '/ssd/xulifeng/workspace/hd_densebox/train/model/net1_iter_150000.caffemodel'
    net = caffe.Net(deploy,caffemodel,caffe.TEST)
    caffe.set_device(0)
    caffe.set_mode_gpu()

    print("listing images...")
    neg_imgs = list_all_files(neg_img_dir)
    neg_idx = 0
    
    print("start detection...")
    for idx, line in enumerate(neg_imgs):
        if idx % 1000 == 0:
            print("%d processed,%d images picked"%(idx, neg_idx))
        img = cv2.imread(line)
        if img is None:
            continue

        score = net_predict(net, img)
        if score >= 0.8:
            sub_folder = os.path.join(out_dir, str(neg_idx/1000))
            mkdir(sub_folder)
            save_file = os.path.join(sub_folder, "%s.jpg"%neg_idx)
            cv2.imwrite(save_file, img)
            fout.write(save_file + "," + line + "\n")
            neg_idx += 1
    fout.close()

def delete_hard_negatives():
    neg_img_dir = "/ssd/xulifeng/workspace/hd_densebox/tmp"
    deploy = '/ssd/xulifeng/workspace/hd_densebox/train/net1_deploy.prototxt'
    caffemodel = '/ssd/xulifeng/workspace/hd_densebox/train/model/net1_iter_1000000.caffemodel'
    net = caffe.Net(deploy,caffemodel,caffe.TEST)
    caffe.set_device(0)
    caffe.set_mode_gpu()

    print("listing images...")
    neg_imgs = list_all_files(neg_img_dir)
    neg_idx = 0
    
    print("start detection...")
    for line in neg_imgs:
        img = cv2.imread(line)
        if img is None:
            continue
        
        score = net_predict(net, img)
        if score > 0.8:
            os.remove(line)
            print("remove:" + line)

# list all files
def list_all_files(dir_name, exts = ['jpg', 'bmp', 'png']):
    result = []
    for dir, sub_dirs, file_names in os.walk(dir_name):
        for file_name in file_names:
            if any(file_name.endswith(ext) for ext in exts):
                result.append(os.path.join(dir, file_name))
    return result
    
def delete_bad_samples():   
    for line in open('/ssd/xulifeng/workspace/hd_densebox/tmp/picked.txt', 'r'):
        words = line.strip().split(",")
        if len(words) < 2:
            continue
        imgName = words[0]
        srcImgName = words[1]
        if os.path.exists(imgName):
            os.remove(srcImgName)
            print("remove:" + srcImgName)

if __name__ == "__main__":
    #collect_hard_negatives()
    delete_bad_samples()
    pass
