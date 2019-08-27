import os
import cv2
import numpy as np
import argparse
from pascal_voc import PascalVocAnn

def list_all_files(dir_name, exts = ['jpg', 'bmp', 'png', 'xml']):
    result = []
    for dir, sub_dirs, file_names in os.walk(dir_name):
        for file_name in file_names:
            if any(file_name.endswith(ext) for ext in exts):
                result.append(os.path.join(dir, file_name))
    return result

def draw_bbox_on_image(nxmin,nymin,nxmax,nymax, img_filepath):
    img = cv2.imread(img_filepath)
    region_rect = cv2.rectangle(img, (int(nxmin),int(nymin)), (int(nxmax),int(nymax)), (0,0,255), 2)
    rect_img_savepath = "example/gt_and_pred/"+os.path.basename(img_filepath)+"gt_and_pred.jpg"
    cv2.imwrite(rect_img_savepath, region_rect)

def draw_by_imgxml(imgpath, xmlpath, img_out):
    _img_ph = imgpath
    _xml_ph = xmlpath
    print("imgpath:%s,  xmlpath:%s"%(imgpath, xmlpath))
    if not os.path.exists(_img_ph):
        raise Exception("%s not exists"%(_img_ph))

    if not os.path.exists(_xml_ph):
        raise Exception("%s not exists"%(_xml_ph))

    #img_ph = "example/originpic/ch00007_20190319_ch00007_20190319151500.mp4.cut.mp4_003000.jpg"
    #xml_ph = "example/originpic/ch00007_20190319_ch00007_20190319151500.mp4.cut.mp4_003000.xml"
    pascal_voc_ann = PascalVocAnn(xml=_xml_ph)
    gtboxes = pascal_voc_ann.get_boxes()
    print("basename:", os.path.basename(_img_ph))
    print("splitext:", os.path.splitext(_img_ph))
    image_basename = os.path.basename(_img_ph)
    img = cv2.imread(_img_ph)
    for bx in gtboxes:
        img = cv2.rectangle(img, (int(bx[1]),int(bx[2])), (int(bx[3]),int(bx[4])), (0,0,255), 1)
    rect_img_savepath = img_out+"/"+os.path.splitext(image_basename)[0]+"_gt.jpg"
    cv2.imwrite(rect_img_savepath, img)   

if __name__ =='__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--imgdir', type=str)
    parse.add_argument('--outdir', type=str)
    _args = parse.parse_args()
    inputimgdir = _args.imgdir;
    imgout = _args.outdir;
    print("imgdir:%s"%(_args.imgdir)) 
    if imgout is not None:
        if not os.path.exists(imgout):
            os.mkdir(imgout)

    inputimgs = []
    print('list all jpgs ...')
    inputimgs.extend(list_all_files(inputimgdir, exts = ['jpg', 'bmp', 'png']))
    print('... done')

    print('draw gt for img...')
    for imgpath in inputimgs:
        sample_basename = os.path.basename(imgpath)
        xml_temp = os.path.splitext(sample_basename)[0]
        xml_name = xml_temp.split('_')[:-1]
        xml_name = '_'.join(xml_name)
        xml_sample_path = "/ssd/hnren/tf/1sd/caffe/data/5_head/HeadVocFormat/FID_DID_HEAD_CLEAN_0_patches_int/Annotations/"+xml_name+".xml"
        draw_by_imgxml(imgpath, xml_sample_path, imgout)
    print('... done')
