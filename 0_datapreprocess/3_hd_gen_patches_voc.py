import os, sys
import cv2, shutil
import numpy as np
import numpy.random as npr
import random
import math
import json
from pascal_voc import PascalVocAnn

#from myAffine import *
#from drawShape import *
            
def distance(pt1, pt2):
    return math.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)

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
    
    cx += (right_dist - left_dist) / (right_dist + left_dist) * fsize * 0.25

    # Transform (cx, cy) back to the original image
    inv_warp_mat = inverseMatrix(warp_mat)
    srcpt = np.array([cx, cy], np.float32)
    dstpt = np.array([0,0], np.float32)
    Affine_Point(inv_warp_mat, srcpt, dstpt)
    cx = dstpt[0]
    cy = dstpt[1]

    return (cx, cy, fsize, alpha)
    
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

def concat_files(in_filenames, out_filename):
    with open(out_filename, 'w') as outfile:
        for fname in in_filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
REQ_IMGSIZE=300#128
IM_RUS=REQ_IMGSIZE/2
def crop_and_resize(img_path, ori_xml, newvocdir):
    if not os.path.exists(newvocdir):
        os.mkdir(newvocdir)
    njpegpth = newvocdir+"/JPEGImages"
    if not os.path.exists(njpegpth):
        os.mkdir(njpegpth)
    nannopth = newvocdir+"/Annotations"
    if not os.path.exists(nannopth):
        os.mkdir(nannopth)
    
    img_filepath = img_path
    lb_filepath = ori_xml
    img = cv2.imread(img_filepath)
    pascal_voc_ann = PascalVocAnn(xml=lb_filepath)
    bboxes = pascal_voc_ann.get_boxes()
    vec = []
    for i,b in enumerate(bboxes):
        print('reshape oribbox to retangle ...')
        xmin, ymin, xmax, ymax = b[1:5]
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        sz = int(max(w, h) * 0.62)
        x = int(xmin + (w - sz) * 0.5)
        y = int(ymin + h - sz)
        vec.extend([x, y, sz, sz])
        xc = x+sz/2
        yc = y+sz/2
        print('crop img with 300x300 ...')
        img_xmn = xc-IM_RUS if xc-IM_RUS>=0 else 0
        img_ymn = yc-IM_RUS if yc-IM_RUS>=0 else 0
        img_xmx = xc+IM_RUS if xc+IM_RUS<img.shape[1] else img.shape[1] - 1
        img_ymx = yc+IM_RUS if yc-IM_RUS<img.shape[0] else img.shape[0] - 1
        ioregion = img[img_ymn:img_ymx,img_xmn:img_xmx]
        #print(img_filepath.split('.'))
        #print(os.path.splitext(img_filepath))
        new_anns_folder = os.path.join(newvocdir, "Annotations")
        new_imgs_folder = os.path.join(newvocdir, "JPEGImages")
        crop_img_name = os.path.splitext(img_filepath)[0]+"_crop_%d.jpg"%(i)
        crop_img_savepath = new_imgs_folder+"/"+os.path.basename(crop_img_name)
        cv2.imwrite(crop_img_savepath, ioregion)
        print('rebase ori bbox pos ...')
        #print("crop_img_savepath:%s", crop_img_savepath)
        nxmin = x- img_xmn
        nymin = y- img_ymn
        nxmax = x+sz-1 - img_xmn
        nymax = y+sz-1 - img_ymn 
        crop_xml_name = os.path.splitext(os.path.basename(crop_img_savepath))[0]+".xml"
        crop_xml_savepath = new_anns_folder+"/"+crop_xml_name
        newpascal_ann = PascalVocAnn(image_name=crop_img_savepath)
        newpascal_ann.set_filename(file_name=crop_img_savepath)
        newpascal_ann.set_size(size=[REQ_IMGSIZE, REQ_IMGSIZE, img.shape[2]])
        newpascal_ann.add_object(object_class="head", xmin=nxmin, ymin=nymin, xmax=nxmax, ymax=nymax)
        newpascal_ann.write_xml(crop_xml_savepath)
        print('... done')

def gen_patches_voc2voc_format(dataset_list, req_imgsize=300):
    anno_type = 1  # fully labelled
    for data_folder in dataset_list:
        ori_anns_folder = os.path.join(data_folder, "Annotations")
        ori_imgs_folder = os.path.join(data_folder, "JPEGImages")
        imgs = list_all_files(ori_imgs_folder, exts=["jpg"])
        newvocdir = data_folder+"_patches"
        print("....crop_and_resize")
        for i, img_path in enumerate(imgs):
            img_base_name = os.path.basename(img_path)
            xml_base_name = os.path.splitext(img_base_name)[0] + ".xml"
            ori_xml = os.path.join(ori_anns_folder, xml_base_name)
            #print("img_base_name>>>> %s, xml_base_name>>>> %s, img_path>>>> %s, ori_xml>>>> %s" %(img_base_name, xml_base_name, img_path, ori_xml))
            crop_and_resize(img_path, ori_xml, newvocdir)

   
def gen_positive_list_voc_format():
    out_file = "/ssd/xulifeng/workspace/hd_densebox/train_data/fid_fullframe_lst.txt"
    dataset_list = ['/ssd/xulifeng/train_data/head_detection/fid/HeadVocFormat/FID_DID_HEAD_CLEAN_0', 
                    '/ssd/xulifeng/train_data/head_detection/fid/HeadVocFormat/FID_DID_HEAD_CLEAN_1', 
                    '/ssd/xulifeng/train_data/head_detection/fid/HeadVocFormat/FID_DID_HEAD_CLEAN_2',
                    '/ssd/xulifeng/train_data/head_detection/fid/HeadVocFormat/HeadBoxDataFidChecked2']
    
    anno_type = 1  # fully labelled
    fout = open(out_file, "wt")
    for data_folder in dataset_list:
        ori_anns_folder = os.path.join(data_folder, "Annotations")
        ori_imgs_folder = os.path.join(data_folder, "JPEGImages")
        imgs = list_all_files(ori_imgs_folder, exts=["jpg"])
        for i, img_path in enumerate(imgs):
            img_base_name = os.path.basename(img_path)
            xml_base_name = os.path.splitext(img_base_name)[0] + ".xml"
            ori_xml = os.path.join(ori_anns_folder, xml_base_name)
            pascal_voc_ann = PascalVocAnn(xml=ori_xml)
            boxes = pascal_voc_ann.get_boxes()            
            if len(boxes) == 0:
                continue

            vec = []
            for b in boxes:
                xmin, ymin, xmax, ymax = b[1:5]
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                sz = int(max(w, h) * 0.62)
                x = int(xmin + (w - sz) * 0.5)
                y = int(ymin + h - sz)
                vec.extend([x, y, sz, sz])

            line = img_path + "," + str(anno_type) + "," + ",".join(map(str, vec)) + "\n"
            fout.write(line)
    fout.close()
    
def gen_positive_list_format1():
    out_file = "/ssd/xulifeng/workspace/hd_densebox/train_data/did_full_all_20190703.txt"
    dataset_list = [('/ssd/xulifeng/train_data/head_detection/did/did_20190609/part0_label.json', '/ssd/xulifeng/train_data/head_detection/did/did_20190609/part0/images'),
                    ('/ssd/xulifeng/train_data/head_detection/did/did_20190609/part1_label.json', '/ssd/xulifeng/train_data/head_detection/did/did_20190609/part1/images')]
    
    '''
    root_dir = "/ssd/xieqiang/Data/nav_tracking_benchmark"
    dataset2 = ['20190703_nav_tracking_head_airport_to_label_0',
                '20190703_nav_tracking_head_to_label_0',
                '20190703_nav_tracking_head_to_label_1',
                '20190703_nav_tracking_head_to_label_2', 
                '20190703_nav_tracking_head_to_label_3']
    for d in dataset2:
        folder = os.path.join(root_dir, d)
        dataset_list.append((os.path.join(folder,"output.json"), os.path.join(folder,"images")))
    '''

    anno_type = 1  # fully labelled
    fout = open(out_file, "wt")
    for (json_file, img_folder) in dataset_list:
        data = json.load(open(json_file, 'r'))
        for image_name in data.keys():
            boxes = data[image_name]
            if len(boxes) == 0:
                continue
            img_path = os.path.join(img_folder, image_name)
            vec = []
            for b in boxes:
                sz = int(max(b[2], b[3]) * 0.62)
                x = int(b[0] + (b[2] - sz) * 0.5)
                y = int(b[1] + b[3] - sz)                    
                vec.extend([x, y, sz, sz])

            line = img_path + "," + str(anno_type) + "," + ",".join(map(str, vec)) + "\n"
            fout.write(line)
    fout.close()
                

def gen_positive_list_format2():
    out_file = "/ssd/xulifeng/workspace/hd_densebox/train_data/head_crops_lst.txt"
    img_roots = [(0, '/ssd/xulifeng/misc/hd_trainset')]
    
    fout = open(out_file, "wt")
    for (anno_type, img_root) in img_roots:
        imgs = list_all_files(img_root, exts = ['jpg', 'bmp', 'png'])
    
        for idx, img_path in enumerate(imgs):
            anno_file = os.path.splitext(img_path)[0] + ".head.txt"
            if not os.path.exists(anno_file):
                continue
            boxes = []
            for line in open(anno_file):
                b = map(int, line.strip().split(","))
                if len(b) < 4:
                    continue
                
                sz = int(max(b[2], b[3]) * 0.62)
                x = int(b[0] + (b[2] - sz) * 0.5)
                y = int(b[1] + b[3] - sz)                    
                boxes.extend([x, y, sz, sz])
            if len(boxes) < 4:
                continue
            line = img_path + "," + str(anno_type) + "," + ",".join(map(str, boxes)) + "\n"
            fout.write(line)
    fout.close()        


def gen_positive_list_from_landmarks():
    out_file = "/ssd/xulifeng/workspace/hd_densebox/train_data/face_crops_lst.txt"
    img_roots = ['/ssd/xulifeng/misc/fd_trainset/part1']
    anno_type = 0
    
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
            b = get_bbox_from_ldmk(ldmk)[0:4]
            x = int(b[0] - (b[2] - 1) * 0.5)
            y = int(b[1] - (b[2] - 1) * 0.5)
            sz = int(b[2])
            img_box_lst.append((img_path, [x, y, sz, sz]))
    
    with open(out_file, "wt") as fout:
        for (img_path, boxes) in img_box_lst:
            line = img_path + "," + str(anno_type) + "," + ",".join(map(str, boxes)) + "\n"
            fout.write(line)
    
def gen_all_positive_list():
    positives = ["/ssd/xulifeng/workspace/hd_densebox/train_data/face_crops_lst.txt",
                 "/ssd/xulifeng/workspace/hd_densebox/train_data/head_crops_lst.txt",
                 #"/ssd/xulifeng/workspace/hd_densebox/train_data/head_did_lst.txt"
                 ]
    workdir = "/ssd/xulifeng/workspace/hd_densebox/train_data"            
    concat_files(positives, os.path.join(workdir, "positive_lst.txt"))

def gen_all_positive_list2():
    positives = ["/ssd/xulifeng/workspace/hd_densebox/train_data/did_fullframe_lst.txt",
                 "/ssd/xulifeng/workspace/hd_densebox/train_data/fid_fullframe_lst.txt",
                 ]
    workdir = "/ssd/xulifeng/workspace/hd_densebox/train_data"            
    concat_files(positives, os.path.join(workdir, "all_fullframe_lst.txt"))
    
if __name__ == "__main__":
    #gen_positive_list_voc_format()
    #gen_positive_list_format1()
    #gen_positive_list_format2()
    #gen_positive_list_from_landmarks()
    #gen_all_positive_list2()
    '''dataset_list = ['/ssd/xulifeng/train_data/head_detection/fid/HeadVocFormat/FID_DID_HEAD_CLEAN_0', 
                    '/ssd/xulifeng/train_data/head_detection/fid/HeadVocFormat/FID_DID_HEAD_CLEAN_1', 
                    '/ssd/xulifeng/train_data/head_detection/fid/HeadVocFormat/FID_DID_HEAD_CLEAN_2',
                    '/ssd/xulifeng/train_data/head_detection/fid/HeadVocFormat/HeadBoxDataFidChecked2']
    '''
    dslist= ['HeadVocFormat/FID_DID_HEAD_CLEAN_0']
 
    gen_patches_voc2voc_format(dataset_list = dslist, req_imgsize=300)
    
    pass
