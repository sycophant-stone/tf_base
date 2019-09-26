import os, sys
import cv2, shutil
import numpy as np
import numpy.random as npr
import random
import math
import json

from myAffine import *
from drawShape import *
            
def distance(pt1, pt2):
    return math.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)

def decide_face_type(pts):
    # 0: frontal face
    # 1: left profile
    # 2: right profile
    face_type = 0
    if (abs(pts[13*2]+1)<0.01 and abs(pts[13*2+1]+1)<0.01):
        face_type = 1  # left profile faces
    elif (abs(pts[34*2]+1)<0.01 and abs(pts[34*2+1]+1)<0.01):
        face_type = 2  # right profile
    else:
        eye1 =[pts[13*2],pts[13*2+1]]
        eye2 =[pts[34*2],pts[34*2+1]]
        p1 = [pts[0],pts[1]]  #left profile
        p2 = [pts[12*2],pts[12*2+1]]
        face_size = distance(p1, p2)
        eye_dist = distance(eye1, eye2)
        dist1 = distance(p1, eye1)
        dist2 = distance(p2, eye2)
        if eye_dist >= 0.6 * face_size:
            face_type = 0
        elif dist1 < dist2:
            face_type = 1
        else:
            face_type = 2

    return face_type
    
def get_bbox_from_ldmk(pts):
    nose =[(pts[47*2] + pts[56*2]) / 2, (pts[47*2+1] + pts[56*2 + 1]) / 2 ] # nose top
    chin =[pts[6*2],pts[6*2+1]] #chin
    alpha = math.atan2(nose[1]-chin[1],nose[0]-chin[0]) + (3.14159 / 2)

    #affine warping the face
    srcCenter = np.array([(nose[0]+chin[0])/2.0,(nose[1]+chin[1])/2.0], np.float32)
    dstCenter = np.array([200,200],np.float32)
    
    scale = 1
    warp_mat = Get_Affine_matrix(dstCenter,srcCenter,alpha,scale)
    
    min_x = 100000
    min_y = 100000
    max_x = -100000
    max_y = -100000
    for n in range(len(pts)/2):
        srcpt = np.array([pts[2*n],pts[2*n+1]], np.float32)
        dstpt = np.array([0,0], np.float32)
        if srcpt[0]!=-1 and srcpt[1]!=-1:
            Affine_Point(warp_mat,srcpt,dstpt)
            if min_x > dstpt[0]:
                min_x = dstpt[0]
            if min_y > dstpt[1]:
                min_y = dstpt[1]
            if max_x < dstpt[0]:
                max_x = dstpt[0]
            if max_y < dstpt[1]:
                max_y = dstpt[1]
        else:
            dstpt[0] = -1
            dstpt[1] = -1

    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    fw = max_x - min_x + 1
    fh = max_y - min_y + 1
    fsize = max(fh, fw)
    
    # adjust face center for profile faces    
    leftpt = [pts[0],pts[1]]  #left profile
    rightpt = [pts[12*2],pts[12*2+1]]
    left_dist = 0
    right_dist = 0
    if (pts[13*2] >= 0 and pts[13*2+1] >= 0):
        left_dist = distance(leftpt, nose)
    if (pts[34*2] >= 0 and pts[34*2+1] >= 0):
        right_dist = distance(rightpt, nose)
    
    if (right_dist + left_dist > 0):
        cx += (right_dist - left_dist) / (right_dist + left_dist) * fsize * 0.25

    # Transform (cx, cy) back to the original image
    inv_warp_mat = inverseMatrix(warp_mat)
    srcpt = np.array([cx, cy], np.float32)
    dstpt = np.array([0,0], np.float32)
    Affine_Point(inv_warp_mat, srcpt, dstpt)
    cx = dstpt[0]
    cy = dstpt[1]

    face_type = decide_face_type(pts)
    
    return (cx, cy, fsize, alpha, face_type)


def write_face_image(face_image, out_imagesdir, crop_idx):
    sub_folder = os.path.join(out_imagesdir, str(crop_idx/1000))
    mkdir(sub_folder)
    outfile = os.path.join(sub_folder, str(crop_idx) + ".jpg")                     
    cv2.imwrite(outfile, face_image)
    return outfile

def mkdir(dr):
    if not os.path.exists(dr):
        os.makedirs(dr)
        
def read_all(file_names):
    lines = []
    for file_name in file_names:
        with open(file_name,'rt') as f:
            lines.extend(f.readlines())
    return lines

# list all image files
def list_all_files(dir_name, exts = ['jpg', 'bmp', 'png']):
    result = []
    for dir, sub_dirs, file_names in os.walk(dir_name):
        for file_name in file_names:
            if any(file_name.endswith(ext) for ext in exts):
                result.append(os.path.join(dir, file_name))
    return result
    
#################################################################
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

def imcrop(img, bbox):
    x1, y1, x2, y2 = bbox
    x2 += 1
    y2 += 1
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        nimg, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
        return nimg[y1:y2, x1:x2, :]
    else:
        return img[y1:y2, x1:x2, :]

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    pad_top = np.abs(np.minimum(0, y1))
    pad_bottom = np.maximum(y2 - img.shape[0], 0)
    pad_left = np.abs(np.minimum(0, x1))
    pad_right = np.maximum(x2 - img.shape[1], 0)

    nimg = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0,0)), mode="constant")
    y1 += pad_top
    y2 += pad_top
    x1 += pad_left
    x2 += pad_left
    return nimg, x1, x2, y1, y2

def enlarge_bbox(bbox, enlarge_ratio):
    x1, y1, x2, y2 = bbox
    x_delta = (x2 - x1 + 1) * (enlarge_ratio - 1) * 0.5
    y_delta = (y2 - y1 + 1) * (enlarge_ratio - 1) * 0.5
    x1 = int(x1 - x_delta)
    x2 = int(x2 + x_delta)
    y1 = int(y1 - y_delta)
    y2 = int(y2 + y_delta)
    return (x1, y1, x2, y2)

def gen_random_crop_boxes(bbox, num, min_iou=0.4, max_iou=1.0, scale_ratio=0.9, shift_ratio=0.3):
    x1, y1, x2, y2 = bbox
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    boxes = []
    while (len(boxes) < num):
        sz = npr.randint(int(min(w, h) * scale_ratio), np.ceil((1/scale_ratio) * max(w, h)))
        # delta here is the offset of box center
        delta_x = npr.randint(-w * shift_ratio, w * shift_ratio + 1)
        delta_y = npr.randint(-h * shift_ratio, h * shift_ratio + 1)
        nx1 = x1 + w * 0.5 + delta_x - sz * 0.5
        ny1 = y1 + h * 0.5 + delta_y - sz * 0.5
        nx2 = nx1 + sz - 1
        ny2 = ny1 + sz - 1
        crop_box = (nx1, ny1, nx2, ny2)
        iou = IoU(crop_box, bbox)
        if iou >= min_iou and iou <= max_iou:
            boxes.append(crop_box)
    return boxes
    
def positives_gen_gt_box(bbox, crop_box, roll_angle, stdsize, enlarge_ratio):
    # region: (Px, Py, Pw, Ph), (Px, Py) is region center
    # ground_truth: (Gx, Gy, Gw, Gh)
    # The regression target are:
    # tx = (Gx - Px) / Pw
    # ty = (Gy - Py) / Ph
    # tw = log(Gw / Pw)
    # th = log(Gh / Ph)
    Px = (crop_box[0] + crop_box[2]) * 0.5
    Py = (crop_box[1] + crop_box[3]) * 0.5
    Pw = float(crop_box[2] - crop_box[0] + 1)
    Ph = float(crop_box[3] - crop_box[1] + 1)
    Gx = (bbox[0] + bbox[2]) * 0.5
    Gy = (bbox[1] + bbox[3]) * 0.5
    Gw = bbox[2] - bbox[0] + 1
    Gh = bbox[3] - bbox[1] + 1
    tx = (Gx - Px) / Pw
    ty = (Gy - Py) / Ph
    tw = np.log(Gw / Pw)
    th = np.log(Gh / Ph)

    fsize = stdsize / enlarge_ratio
    gcx = (stdsize - 1) * 0.5 + tx * fsize
    gcy = (stdsize - 1) * 0.5 + ty * fsize
    gw = fsize * np.exp(tw)
    groll = -roll_angle*0.1    # The angle is counter-clockwise rotation, we need clockwise
    return (gcx, gcy, gw, groll)

    
# In this format, all annotations are stored in a file. each image takes one line. Each image has one or more faces
# box = (center_x, center_y, width, angle, face_type)
def load_face_boxes_format1(gt_files):
    img_box_lst = []
    linesoflandmarks = read_all(gt_files)
    N = 144
    
    for idx, line in enumerate(linesoflandmarks):
        boxes = []
        if "," in line:
            words = line.strip().split(",")
        else:
            words = line.strip().split(" ")
        
        if len(words) < N+1:
            continue

        imgname = words[0]
        imgname = imgname.replace('/data/xulifeng', '/ssd/xulifeng/misc')
        pts = map(float, words[1:])
        nface = len(pts) / N
        for i in range(nface):
            boxes.append(get_bbox_from_ldmk(pts[N*i : N*(i+1)]))
        img_box_lst.append((imgname, boxes))
    return img_box_lst

# In this format, each image has an annotation file
# box = (center_x, center_y, width, angle, face_type)
def load_face_boxes_format2(img_roots):
    imgs = []
    for img_root in img_roots:
        imgs.extend(list_all_files(img_root, exts = ['jpg', 'bmp', 'png']))

    img_box_lst = []
    for img_path in imgs:
        anno_file = os.path.splitext(img_path)[0] + ".ldmk.txt"
        if not os.path.exists(anno_file):
            continue
        with open(anno_file) as f:
            ldmk = map(float, next(f).strip().split(","))
            bbox = get_bbox_from_ldmk(ldmk)
            img_box_lst.append((img_path, [bbox]))
    return img_box_lst

# sampling from Multi-PIE
def load_face_boxes_multipie():
    gt_files = ['/ssd/xulifeng/misc/facedataset/multi-pie_ldmks.txt']
    img_box_lst = []
    linesoflandmarks = read_all(gt_files)
    N = 144
    
    for idx, line in enumerate(linesoflandmarks):
        
        # Do sampling
        if idx % 6 != 0:
            continue

        boxes = []
        if "," in line:
            words = line.strip().split(",")
        else:
            words = line.strip().split(" ")
        
        if len(words) < N+1:
            continue

        imgname = words[0]
        
        pts = map(float, words[1:])
        nface = len(pts) / N
        for i in range(nface):
            boxes.append(get_bbox_from_ldmk(pts[N*i : N*(i+1)]))
        img_box_lst.append((imgname, boxes))
    return img_box_lst
    
def test():
    print("loading face boxes...")
    img_roots = ['/ssd/xulifeng/misc/fd_trainset_withcontext/hard']
    all_list = load_face_boxes_format2(img_roots)
    
    for img_idx, line in enumerate(all_list):
        img_name = line[0]
        b = line[1] #bbox
        print("[%d/%d]: "%(img_idx,len(all_list)) + img_name)
        img = cv2.imread(img_name)
        w = b[2] * 0.5
        cv2.rectangle(img,(int(b[0]-w), int(b[1]-w)),(int(b[0]+w), int(b[1]+w)),(255,0,0),1)
        cv2.imwrite(str(img_idx)+".jpg", img)
        if (img_idx >= 30):
            break

def crop_positives_net1(out_imagesdir, stdsize, enlarge_ratio, randRoll, iou, shift):
    mkdir(out_imagesdir)
    mkdir(os.path.join(out_imagesdir, "positive"))
    mkdir(os.path.join(out_imagesdir, "partface"))
    f_positive = open(os.path.join(out_imagesdir, "positive_lst.txt"), 'wt')
    f_partface = open(os.path.join(out_imagesdir, "partface_lst.txt"), 'wt')

    print("loading face boxes...")
    all_image_list = []
    gt_files = ['/ssd/xulifeng/misc/facedataset/image_set/zhu_72pts_select_all_v3.txt',
            #'/ssd/xulifeng/misc/facedataset/image_set/AFLW_Alignment_groundTruth.txt',
            #'/ssd/xulifeng/misc/facedataset/image_set/zhongce_train_72pts_contourModified.txt',
            #'/ssd/xulifeng/misc/facedataset/image_set/zhongce_test_72pts_contourModified.txt'
            ]
    all_image_list.extend(load_face_boxes_format1(gt_files))
    
    img_roots = ['/ssd/xulifeng/misc/fd_trainset_withcontext']
    all_image_list.extend(load_face_boxes_format2(img_roots))
    
    positive_num = 0
    partface_num = 0
    
    print("start cropping...")

    for idx, (img_name, boxes) in enumerate(all_image_list):
        if (idx % 100 == 0):
            print(idx, len(all_image_list))
        
        if (len(boxes) == 0):
            continue
            
        img = cv2.imread(img_name)
        if img is None:
            print('can not load image:' + img_name)
            continue
        
        imgh, imgw, imgc = img.shape
        
        for box in boxes:
            (center_x, center_y, width, angle, face_type) = box[0:5]

            if width < 40:
                continue

            # remove incomplete faces
            tol = int(width * (enlarge_ratio * 0.5 - 0.4))
            if ((center_x - tol < 0) or (center_x + tol >= imgw) or (center_y - tol < 0 ) or (center_y + tol >= imgh)):
                continue

            # random rotate image: Positive values mean counter-clockwise rotation
            roll_angle = npr.randint(-randRoll, randRoll)
            rotate = angle * (180 / 3.14159) + roll_angle

            face_cols = int(1.6 * enlarge_ratio * width)
            face_rows = face_cols
            cx = (face_cols - 1) * 0.5
            cy = (face_rows - 1) * 0.5

            matrix = cv2.getRotationMatrix2D((center_x, center_y), rotate, 1)
            matrix[0, 2] += cx - center_x
            matrix[1, 2] += cy - center_y
            face_img = cv2.warpAffine(img, matrix, (face_cols, face_rows), borderMode=cv2.BORDER_REPLICATE)
            delta = (width - 1) * 0.5
            bbox = (cx-delta, cy-delta, cx+delta, cy+delta)

            for flip_idx in range(0, 2):
                if (flip_idx == 1):
                    face_img = cv2.flip(face_img, 1) # horizontally flip
                    roll_angle = -roll_angle
                
                for crop_type in ["positive", "partface"]:
                    crop_num = 1
                    if crop_type == "positive":
                        crop_boxes = gen_random_crop_boxes(bbox, crop_num, min_iou=iou, max_iou=1.01, scale_ratio=0.88, shift_ratio=shift)
                    else:
                        #crop_boxes = gen_random_crop_boxes(bbox, crop_num, min_iou=0.1, max_iou=iou, scale_ratio=0.88, shift_ratio=shift+0.2)
                        crop_boxes = gen_random_crop_boxes(bbox, crop_num, min_iou=0.03, max_iou=iou, scale_ratio=0.88, shift_ratio=shift+0.2)

                    for crop_box in crop_boxes:
                        cropped_im = imcrop(face_img, enlarge_bbox(crop_box, enlarge_ratio))
                        resized_im = cv2.resize(cropped_im, (stdsize, stdsize), interpolation=cv2.INTER_LINEAR)
                        (gcx, gcy, gw, groll) = positives_gen_gt_box(bbox, crop_box, roll_angle, stdsize, enlarge_ratio)

                        if crop_type == "positive":
                            outfile = os.path.join(out_imagesdir, crop_type, str(positive_num/5000), str(positive_num) + ".png")
                            
                            label = 1                            
                            txtline = outfile + ",%d,%f,%f,%f\n"%(label, gcx, gcy, gw)                          
                            f_positive.write(txtline)
                            positive_num += 1
                        else:
                            outfile = os.path.join(out_imagesdir, crop_type, str(partface_num/5000), str(partface_num) + ".png")
                            
                            label = -1
                            groll = -100  # ignore angle for part samples
                            txtline = outfile + ",%d,%f,%f,%f\n"%(label, gcx, gcy, gw)
                            f_partface.write(txtline)
                            partface_num += 1
                        
                        mkdir(os.path.dirname(outfile))
                        cv2.imwrite(outfile, resized_im) 
    f_positive.close()
    f_partface.close()

   
def gen_net1_data():
    out_imagesdir = '/ssd/xulifeng/workspace/hd_stage1/train_data_net1/positives_add1'
    crop_positives_net1(out_imagesdir, stdsize=49, enlarge_ratio=2.72, randRoll=15, iou = 0.4, shift = 0.25)

def gen_net2_data():
    out_imagesdir = '/ssd/xulifeng/workspace/hd_stage1/train_data_net2/positives_add1'
    crop_positives_net1(out_imagesdir, stdsize=49, enlarge_ratio=4.0, randRoll=15, iou = 0.26, shift = 0.35)

def gen_net3_data():
    out_imagesdir = '/ssd/xulifeng/workspace/hd_stage1/train_data_net3/positives_add1'
    crop_positives_net1(out_imagesdir, stdsize=49, enlarge_ratio=6.0, randRoll=15, iou = 0.14, shift = 0.5)
    
if __name__ == "__main__":
    #gen_net1_data()
    #gen_net2_data()
    gen_net3_data()
    pass
