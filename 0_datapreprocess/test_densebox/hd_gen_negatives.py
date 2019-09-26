import os, sys
import detect_tools as dtools
import cv2
import numpy as np
import numpy.random as npr
import os
import random
import math

def distance(pt1, pt2):
    return math.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)
    
def mkdir(dr):
    if not os.path.exists(dr):
        os.makedirs(dr)

# list all files
def list_all_files(dir_name, exts = ['jpg', 'bmp', 'png']):
    result = []
    for dir, sub_dirs, file_names in os.walk(dir_name):
        for file_name in file_names:
            if any(file_name.endswith(ext) for ext in exts):
                result.append(os.path.join(dir, file_name))
    return result

def detect_by_net(net, img):
    threshold = 0.5
    scale_factor = 0.75
    stride = 4
    label_width = 32
    label_height = 32
    return dtools.detect_multiscale(net, img, threshold, scale_factor, stride, label_width, label_height)

def load_net(device_id = 0):
    net_root = '/ssd/xulifeng/workspace/hd_densebox/train2'
    net_deploy = os.path.join(net_root, 'net1_deploy.prototxt')
    for iter in range(10, 0, -1):
        model_name = 'net1_iter_%d.caffemodel' % (iter * 100000)
        net_caffemodel = os.path.join(net_root, 'model', model_name)
        if os.path.exists(net_caffemodel):
            break
    net = dtools.load_net(net_deploy, net_caffemodel, device_id = device_id)
    return net

def mine_hard_negatives(total_num, device_id):
    net = load_net(device_id)
    neg_save_root = os.path.join("/ssd/xulifeng/workspace/hd_densebox/train_data", "negatives2")

    for dir_idx in range(0, 20):
        neg_save_dir = os.path.join(neg_save_root, "mine_hard"+str(dir_idx))
        if (not os.path.exists(neg_save_dir)):
            mkdir(neg_save_dir)
            break
    
    neg_idx = 0
    rotateDegrees = [0, 180, 270, 90, 0]
    proc_image_num = 0
    
    for rotateDegree in rotateDegrees:
        if (neg_idx >= total_num):
            break
        print("Loading negative imgage list...")
        neg_img_dirs = ["/ssd/xulifeng/negatives/body_crop",
                        "/ssd/xulifeng/negatives/no_human"]

        #if not (rotateDegree == 0):
        #    neg_img_dirs.extend(["/ssd/xulifeng/negatives/only_for_face"])

        neg_imgs = []
        for folder in neg_img_dirs:
            neg_imgs.extend(list_all_files(folder, exts = ['jpg', 'bmp', 'png']))
        random.shuffle(neg_imgs)
        
        for line in neg_imgs:
            if (neg_idx >= total_num):
                break

            img = cv2.imread(line)
            if img is None:
                continue
            # print "processing: " + line
            proc_image_num += 1
            if proc_image_num % 20 == 0:
                print("%d negatives cropped; %d image processed."%(neg_idx, proc_image_num))    
                
            if (rotateDegree == 90):
                img = cv2.flip(cv2.transpose(img), 1)
            elif (rotateDegree == 180):
                img = cv2.flip(img, -1)
            elif (rotateDegree == 270):
                img = cv2.flip(cv2.transpose(img), 0)

            crops = detect_by_net(net, img)

            for cropped_im in crops:
                sub_folder = os.path.join(neg_save_dir, str(neg_idx/2000))
                mkdir(sub_folder)
                save_file = os.path.join(sub_folder, "%s.png"%neg_idx)
                cv2.imwrite(save_file, cropped_im)
                neg_idx += 1


def random_crop_negatives(total_num, stdsize):
    neg_save_root = os.path.join("/ssd/xulifeng/workspace/hd_densebox/train_data", "negatives")
    neg_save_dir = os.path.join(neg_save_root, "random_crop")
    neg_img_dir = "/ssd/xulifeng/negatives/no_human"
    mkdir(neg_save_dir)

    neg_imgs = list_all_files(neg_img_dir, exts = ['jpg', 'bmp', 'png'])
    random.shuffle(neg_imgs)
    neg_idx = 0

    while (True):
        if (neg_idx >= total_num):
            break

        for line in neg_imgs:
            if (neg_idx >= total_num):
                break

            img = cv2.imread(line)
            if img is None:
                continue

            height, width, channel = img.shape

            local_num = 0
            while local_num < 5:
                minsize = stdsize - 1
                size = npr.randint(minsize, max(minsize + 2, min(width, height) / 2))
                if width <= size or height <= size:
                    break
                nx = npr.randint(0, width - size)
                ny = npr.randint(0, height - size)
                cropped_im = img[ny : ny + size, nx : nx + size, :]
                resized_im = cv2.resize(cropped_im, (stdsize, stdsize), interpolation=cv2.INTER_LINEAR)

                sub_folder = os.path.join(neg_save_dir, str(neg_idx/5000))
                mkdir(sub_folder)
                save_file = os.path.join(sub_folder, "%s.png"%neg_idx)
                cv2.imwrite(save_file, resized_im)
                neg_idx += 1
                local_num += 1
                if neg_idx % 1000 == 0:
                    print neg_idx, " images cropped"

def IoU(box, gt_box):
    """Compute IoU between detect box and gt box
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)
    xx1 = np.maximum(box[0], gt_box[0])
    yy1 = np.maximum(box[1], gt_box[1])
    xx2 = np.minimum(box[2], gt_box[2])
    yy2 = np.minimum(box[3], gt_box[3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = float(inter) / float(box_area + area - inter)
    return ovr


def gen_negative_list():
    workdir = "/ssd/xulifeng/workspace/hd_densebox/train_data"
    negatives_imgset = os.path.join(workdir, "negatives2")
    out_negative_file = os.path.join(workdir, "negative2_lst.txt")
    
    lines = list_all_files(negatives_imgset, exts = ['jpg', 'bmp', 'png'])
    print "negatives = ", len(lines)
    
    f = open(out_negative_file, "wt")
    for line in lines:
        f.write(line + ",1\n")
    f.close()

    
def gen_data_for_net1():
    #random_crop_negatives(50000, stdsize=96)
    #mine_hard_negatives(50000, device_id = 0)
    gen_negative_list()

if __name__ == '__main__':
    gen_data_for_net1()
    pass
    
