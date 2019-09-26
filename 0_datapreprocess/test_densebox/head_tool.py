import os, sys
import cv2, shutil
import numpy as np
import numpy.random as npr
import random
import math
import json
import csv

from myAffine import *
            
def distance(pt1, pt2):
    return math.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)

def decide_face_type(pts):
    # 0: frontal face
    # 1: left profile
    # 2: right profile
    face_type = 0
    facing_toward = 0
    if (abs(pts[13*2]+1)<0.01 and abs(pts[13*2+1]+1)<0.01):
        face_type = 1  # left profile faces
        facing_toward = 1
    elif (abs(pts[34*2]+1)<0.01 and abs(pts[34*2+1]+1)<0.01):
        face_type = 2  # right profile
        facing_toward = 2
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
            
        if dist1 < dist2:
            facing_toward = 1
        else:
            facing_toward = 2

    return (face_type, facing_toward)

def getBox(pts):
    (face_type, facing_toward) = decide_face_type(pts)
    pt1 =[(pts[47*2] + pts[56*2]) / 2, (pts[47*2+1] + pts[56*2 + 1]) / 2 ] # nose top
    pt2 =[pts[6*2],pts[6*2+1]] #chin
    alpha = math.atan2(pt1[1]-pt2[1],pt1[0]-pt2[0]) + (3.14159 / 2)

    #affine warping the face
    srcCenter = np.array([(pt1[0]+pt2[0])/2.0,(pt1[1]+pt2[1])/2.0], np.float32)
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
    shift = (fh - fw) * 0.7
    if (face_type == 1):
        cx += shift
    elif (face_type == 2):
        cx -= shift
    
    # Transform (cx, cy) back to the original image
    inv_warp_mat = inverseMatrix(warp_mat)
    srcpt = np.array([cx, cy], np.float32)
    dstpt = np.array([0,0], np.float32)
    Affine_Point(inv_warp_mat, srcpt, dstpt)
    cx = dstpt[0]
    cy = dstpt[1]

    return (cx, cy, fsize, alpha, face_type)


def load_face_boxes(line):
    boxes = []
    ldmks = []
    if "," in line:
        words = line.strip().split(",")
    else:
        words = line.strip().split(" ")
    
    if len(words) < 145:
        return (None, None, None)

    imgName = words[0]
    pts = map(float, words[1:])
    nface = len(pts) / 144
    for i in range(nface):
        ldmk = pts[144*i : 144*(i+1)]
        boxes.append(getBox(ldmk))
        ldmks.append(ldmk)
    return (imgName, boxes, ldmks)


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
    
def crop_heads_from_landmarks():
    gt_files = ['/data/xulifeng/facedataset/image_set/sunglasses_landmarks.txt',
                '/data/xulifeng/facedataset/image_set/zhu_72pts_select_all_v3.txt',
                ]
    out_imagesdir = '/data/xulifeng/headcrop2/face'
    mkdir(out_imagesdir)
    
    linesoflandmarks = read_all(gt_files)
    crop_idx = 0
    
    for line in linesoflandmarks:
        (imgname, boxes, ldmks) = load_face_boxes(line)
        if boxes is None:
            continue
        
        if(len(boxes) > 10):
            continue
        
        print("proc: ", imgname)
        img = cv2.imread(imgname)
        if img is None :
            print 'can not load image:', imgname
            continue

        imgh, imgw, imgc = img.shape
        for box_idx, box in enumerate(boxes):
            (center_x, center_y, width, angle, face_type) = box

            # skip small faces
            if (width < 45):
                continue
            
            # skip incomplete heads
            tol = width * 2
            if ((center_x - tol < 0) or (center_x + tol >= imgw) or (center_y - tol < 0 ) or (center_y + tol*1.3 >= imgh)):
                continue

            # skip large-angle faces (> 45 degrees)
            if (abs(angle) >= 3.14159 / 4):
                continue
                
            rotate = 0
            enlargeRatio = 6.5
            CROP_WIDTH = 360
            CROP_HEIGHT = 360
            dcx = (CROP_WIDTH - 1) * 0.5
            dcy = (CROP_HEIGHT - 1) * 0.4
            scale = CROP_WIDTH / (enlargeRatio * width);            
            matrix = cv2.getRotationMatrix2D((center_x, center_y), rotate, scale)
            matrix[0, 2] += dcx - center_x
            matrix[1, 2] += dcy - center_y
            cropped = cv2.warpAffine(img, matrix, (CROP_WIDTH, CROP_HEIGHT), borderMode=cv2.BORDER_REPLICATE)
            
            ldmk = ldmks[box_idx]
            cropped_ldmk = ldmk[:]
            for i in range(len(cropped_ldmk)/2):
                x,y= cropped_ldmk[i*2: (i+1)*2]
                if x >= 0 and y >= 0:
                    cropped_ldmk[i*2] = int((cropped_ldmk[i*2] - center_x) * scale + dcx)
                    cropped_ldmk[i*2+1] = int((cropped_ldmk[i*2+1] - center_y) * scale + dcy)
            
            #for i in range(len(cropped_ldmk)/2):
            #    cv2.circle(cropped, (cropped_ldmk[i*2],cropped_ldmk[i*2+1]), 2, (0,0,255),-1)
            
            sub_folder = os.path.join(out_imagesdir, str(crop_idx/1000))
            mkdir(sub_folder)
            outfile = os.path.join(sub_folder, str(crop_idx) + ".jpg")
            cv2.imwrite(outfile, cropped) 

            outfile = outfile.replace(".jpg", ".ldmk.txt")
            with open(outfile, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(cropped_ldmk)            
            crop_idx = crop_idx + 1         
        #if (crop_idx > 200):
        #    break
        

def load_head_boxes_from_haoxiang_coco():
    dataset_list = [#('/mnt/soulfs2/zq/head_detection_training/head_labeling_norm_0916_coco.json', '/mnt/soulfs2/zq/head_detection_training/head_labeling_norm_0916/images'),
                    ('/mnt/soulfs2/lihaoxiang/detection-train/XHM_20181021/1004-1005_regular_2min_head.coco', ''),
                    ('/mnt/soulfs2/lihaoxiang/detection-train/DFXTD_20181017/regular_1004_1005_2min_head_train.coco', '')
                   ]
    all_list = []
    for dataset in dataset_list:
        image_folder = dataset[1]
        data = json.load(open(dataset[0], 'r'))
        names_ids = [(i['file_name'], i['id']) for i in data['images']]
        id_boxes = {}
        for image_name, image_id in names_ids:  
            id_boxes[image_id] = []
        for i in data['annotations']:
            b = i['bbox']
            image_id = i['image_id']
            id_boxes[image_id].append(b)
            
        for image_name, image_id in names_ids:        
            img_path = os.path.join(image_folder, image_name)
            all_list.append((img_path, id_boxes[image_id]))
    return all_list

def crop_heads_from_haoxiang_coco():
    out_imagesdir = '/data/xulifeng/headcrop2/head_bbox'
    mkdir(out_imagesdir)    
    print("loading head boxes")
    all_list = load_head_boxes_from_haoxiang_coco()
    
    print("Run cropping")
    crop_idx = 0
    
    for img_idx, (img_path, boxes) in enumerate(all_list):       
        print("proc: ", img_path)
        img = cv2.imread(img_path)
        if img is None :
            print('can not load image:', img_path)
            continue
        imgh, imgw, imgc = img.shape
        
        for b in boxes:
            # skip occluded head
            if (b[2] > (b[3] * 1.2)):
                continue
                
            # skip incomplete heads
            tol = b[2] * 2
            if ((b[0] - tol < 0) or (b[0]+b[2] + tol >= imgw) or (b[1] - tol < 0 ) or (b[1]+b[3] + tol*1.3 >= imgh)):
                continue
           
            width = max(b[2], b[3]) * 0.56
            
            # skip small heads
            if (width < 30):
                continue
            
            center_x = b[0] + b[2] * 0.5
            center_y = b[1] + b[3] - (width * 0.6) #the head bottom is not accurate, so I move up a little
            
            rotate = 0
            enlargeRatio = 6.5
            CROP_WIDTH = 360
            CROP_HEIGHT = 360
            dcx = (CROP_WIDTH - 1) * 0.5
            dcy = (CROP_HEIGHT - 1) * 0.4
            scale = CROP_WIDTH / (enlargeRatio * width);            
            matrix = cv2.getRotationMatrix2D((center_x, center_y), rotate, scale)
            matrix[0, 2] += dcx - center_x
            matrix[1, 2] += dcy - center_y
            cropped = cv2.warpAffine(img, matrix, (CROP_WIDTH, CROP_HEIGHT), borderMode=cv2.BORDER_REPLICATE)
            new_box = list(b)
            new_box[0] = int((new_box[0] - center_x) * scale + dcx)
            new_box[1] = int((new_box[1] - center_y) * scale + dcy)
            new_box[2] = int(new_box[2] * scale)
            new_box[3] = int(new_box[3] * scale)

            #cv2.rectangle(cropped, (new_box[0], new_box[1]), (new_box[0]+new_box[2], new_box[1]+new_box[3]), (0,0,255))
            
            sub_folder = os.path.join(out_imagesdir, str(crop_idx/1000))
            mkdir(sub_folder)
            outfile = os.path.join(sub_folder, str(crop_idx) + ".jpg")
            cv2.imwrite(outfile, cropped) 

            outfile = outfile.replace(".jpg", ".head.txt")
            with open(outfile, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(new_box)            
            crop_idx = crop_idx + 1

def load_face_bbox():
    in_folder = "/data/xulifeng/fisheye_face_train_tolabel/part0"
    all_list = []
    all_imgs = list_all_files(in_folder, exts = ['jpg', 'bmp', 'png'])
    for img_path in all_imgs:
        gt_path = os.path.splitext(img_path)[0] + ".rect.txt"
        if not os.path.exists(gt_path):
            continue
        boxes = []
        with open(gt_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                boxes.append(map(int, row))
        if len(boxes) == 0:
            continue
        all_list.append((img_path, boxes))
    return all_list
    
def load_face_bbox_json(folder):
    all_list = []
    anno_file_path = os.path.join(folder, "output.json")
    img_root = os.path.join(folder, "images")
    
    with open(anno_file_path,'r') as f:
        data = json.load(f)
    for img_name in data.keys():              
        boxes = []
        for bbox in data[img_name]:
            if "head" in bbox[4]:
                boxes.append(bbox[0:4])
        if len(boxes) == 0:
            continue
        img_path = os.path.join(img_root, img_name)
        all_list.append((img_path, boxes))
    return all_list
    
def headbox_convert_format():
    folder = '/ssd/xulifeng/misc/head_frames_labelled/part2'
    all_list = load_face_bbox_json(folder)
    for (img_path, boxes) in all_list:
        gt_path = os.path.splitext(img_path)[0] + ".head.txt"
        with open(gt_path, 'wt') as f:
            for b in boxes:
                f.write("%d,%d,%d,%d\n"%(b[0], b[1], b[2], b[3]))

def crop_heads_from_facelabeller():
    out_imagesdir = '/data/xulifeng/fisheye_face_train_tolabel/airport_vip'
    mkdir(out_imagesdir)    
    print("loading face boxes")
    all_list = []
    all_list.extend(load_face_bbox_json("/mnt/soulfs2/zq/airport/20181218_qingdao-airport_13_pano"))
    all_list.extend(load_face_bbox_json("/mnt/soulfs2/zq/airport/20181218_qingdao-airport_14_1_pano"))
    
    print("Run cropping")
    crop_idx = 0
    
    for img_idx, (img_path, boxes) in enumerate(all_list):       
        print("proc: ", img_path)
        img = cv2.imread(img_path)
        if img is None :
            print('can not load image:', img_path)
            continue
        imgh, imgw, imgc = img.shape
        
        for b in boxes:
            width = max(b[2], b[3])
            if width < 20:
                continue
            
            center_x = b[0] + b[2] * 0.5
            center_y = b[1] + b[3] * 0.5
            
            rotate = 0
            enlargeRatio = 6.0
            CROP_WIDTH = 360
            CROP_HEIGHT = 360
            scale = CROP_WIDTH / (enlargeRatio * width);
            
            dcx = (CROP_WIDTH - 1) * 0.5
            dcy = (CROP_HEIGHT - 1) * 0.4            
            
            matrix = cv2.getRotationMatrix2D((center_x, center_y), rotate, scale)
            matrix[0, 2] += dcx - center_x
            matrix[1, 2] += dcy - center_y
            cropped = cv2.warpAffine(img, matrix, (CROP_WIDTH, CROP_HEIGHT), borderMode=cv2.BORDER_REPLICATE)
            new_box = list(b)
            new_box[0] = int((new_box[0] - center_x) * scale + dcx)
            new_box[1] = int((new_box[1] - center_y) * scale + dcy)
            new_box[2] = int(new_box[2] * scale)
            new_box[3] = int(new_box[3] * scale)
            
            sub_folder = os.path.join(out_imagesdir, str(crop_idx/2000))
            mkdir(sub_folder)
            outfile = os.path.join(sub_folder, str(crop_idx) + ".jpg")
            cv2.imwrite(outfile, cropped) 

            outfile = outfile.replace(".jpg", ".rect.txt")
            with open(outfile, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(new_box)            
            crop_idx = crop_idx + 1
            
def crop_for_check_bbox():
    img_root = "/data/xulifeng/headcrop2/mall_bbox_1"
    out_root = "/data/xulifeng/headcrop2/tmp"
    mkdir(out_root)
    
    list_file = open(os.path.join(out_root, "imglist.txt"), "wt")
    
    all_imgs = list_all_files(img_root, exts = ['jpg', 'bmp', 'png'])
    for idx, img_name in enumerate(all_imgs):
        gt_name = os.path.splitext(img_name)[0] + ".head.txt"
        if not os.path.exists(gt_name):
            continue
        with open(gt_name) as f:
            b = map(int, next(f).strip().split(","))
        
        img = cv2.imread(img_name)
        if img is None :
            print('can not load image:', img_name)
            continue 
        
        width = max(b[2], b[3]) * 0.56
        center_x = b[0] + b[2] * 0.5
        center_y = b[1] + b[3] - (width * 0.6) #the head bottom is not accurate, so I move up a little        
        rotate = 0
        enlargeRatio = 2.2
        CROP_WIDTH = 128
        CROP_HEIGHT = 128
        dcx = (CROP_WIDTH - 1) * 0.5
        dcy = (CROP_HEIGHT - 1) * 0.5
        scale = CROP_WIDTH / (enlargeRatio * width);            
        matrix = cv2.getRotationMatrix2D((center_x, center_y), rotate, scale)
        matrix[0, 2] += dcx - center_x
        matrix[1, 2] += dcy - center_y
        cropped = cv2.warpAffine(img, matrix, (CROP_WIDTH, CROP_HEIGHT), borderMode=cv2.BORDER_REPLICATE)        
        outfile = os.path.join(out_root, str(idx/2000), str(idx) + ".jpg")
        mkdir(os.path.dirname(outfile))
        print("write", outfile)
        cv2.imwrite(outfile, cropped)
        list_file.write(outfile + "," + img_name + "\n")
    list_file.close()

def delete_bad_samples():
    infile = "/data/xulifeng/headcrop2/tmp/imglist.txt"
    with open(infile, 'rt') as f:
        all_lines = f.readlines()

    for line in all_lines:
        src_path, dst_path = line.strip().split(",")
        if not os.path.exists(src_path):
            try:
                print("del", dst_path)
                os.remove(dst_path)
                os.remove(dst_path.replace(".jpg", ".head.txt"))
            except Exception as e:
                continue

def ldmks_convert_format():
    anno_file_path = "/data/xulifeng/face_crop/ldmks.txt"
    tmp = np.loadtxt(anno_file_path, dtype=np.str, delimiter=" ")
    images = tmp[:,0]
    ldmks = tmp[:,1:1+144].astype(np.int32)
    
    for idx in range(ldmks.shape[0]):
        outfile = images[idx].replace(".jpg", ".ldmk.txt")
        print("write", outfile)
        ldmk = ldmks[idx]
        with open(outfile, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(ldmk)

def batch_rename_files():
    in_folder = "/ssd/xulifeng/misc/head_frames_labelled/part2"
    out_root = '/ssd/xulifeng/misc/head_frames_labelled/part1'
    #anno_ext = ".ldmk.txt"
    anno_ext = ".head.txt"
    mkdir(out_root)
    
    imgs = list_all_files(in_folder, exts = ['jpg', 'bmp', 'png'])
    idx = 0
    for src_name in imgs:
        anno_file = src_name.replace(".jpg", anno_ext)
        if not os.path.exists(anno_file):
            continue
        
        dst_name = os.path.join(out_root, "%d/%06d.jpg"%(idx/1000, idx))
        mkdir(os.path.dirname(dst_name))
        os.rename(src_name, dst_name)
        
        src_name = anno_file
        dst_name = dst_name.replace(".jpg", anno_ext)
        os.rename(src_name, dst_name)
        
        idx += 1
        if (idx % 100 == 0):
            print(idx)
            
if __name__ == "__main__":
    #headbox_convert_format()
    batch_rename_files()
    pass
