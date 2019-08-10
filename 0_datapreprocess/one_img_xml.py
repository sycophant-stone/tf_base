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



if __name__ =='__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--img_ph', type=str)
    parse.add_argument('--xml_ph', type=str)
    parse.add_argument('--env', type=str)
    _args = parse.parse_args()
    _img_ph= _args.img_ph;
    _xml_ph= _args.xml_ph;

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
        img = cv2.rectangle(img, (int(bx[1]),int(bx[2])), (int(bx[3]),int(bx[4])), (0,0,255), 2)
    rect_img_savepath = "./"+os.path.splitext(image_basename)[0]+"_gt.jpg"
    cv2.imwrite(rect_img_savepath, img)
