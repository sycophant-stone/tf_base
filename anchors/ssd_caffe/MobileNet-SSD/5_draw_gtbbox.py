import os
import cv2
import numpy as np
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
    region_rect = cv2.rectangle(img, (int(nxmin),int(nymin)), (int(nxmax),int(nymax)), (0,0,255), 1)
    rect_img_savepath = "example/gt_and_pred_1/"+os.path.basename(img_filepath)+"gt_and_pred.jpg"
    cv2.imwrite(rect_img_savepath, region_rect)



if __name__ =='__main__':
    jpgs = list_all_files("example/res_1", exts=["jpg"])
    for jpg_img in jpgs:
        temp = os.path.basename(jpg_img)#[:-4]+".xml"
        temp = os.path.splitext(temp)[0]
        print(temp[:-4])
        xmlpath = "example/anno/"+temp[:-4]+".xml"
        pascal_voc_ann = PascalVocAnn(xml=xmlpath)
        gtboxes = pascal_voc_ann.get_boxes()
        draw_bbox_on_image(gtboxes[0][1], gtboxes[0][2], gtboxes[0][3], gtboxes[0][4], jpg_img)
