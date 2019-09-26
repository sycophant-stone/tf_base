"""gen pascal annotations according to txt
"""
import os, sys
import cv2, shutil
import numpy as np
import numpy.random as npr
import random
import math
import json
import shell
from pascal_voc import PascalVocAnn



# list all image files
def list_all_files(dir_name, exts = ['jpg', 'bmp', 'png', 'xml', 'txt']):
    result = []
    for dir, sub_dirs, file_names in os.walk(dir_name):
        for file_name in file_names:
            if any(file_name.endswith(ext) for ext in exts):
                result.append(os.path.join(dir, file_name))
    return result


def gen_pascal_xml(image_path, size, box, xml_dst_path):
    newpascal_ann = PascalVocAnn(image_name = image_path)
    newpascal_ann.set_filename(file_name    = image_path)
    newpascal_ann.set_size(size=size)

    nxmin,nymin,nxmax,nymax = box
    newpascal_ann.add_object(object_class="head", xmin=nxmin, ymin=nymin, xmax=nxmax, ymax=nymax)
    newpascal_ann.check_boxes()
    newpascal_ann.write_xml(xml_dst_path)


def gen_pascal_anno(txt_anno_dirlist):
    '''gen pascal anno
    '''

    for anno_dir in txt_anno_dirlist:

        dst_image_path = anno_dir + "/JPEGImages/"
        dst_xml_path   = anno_dir + "/Annotations/"
        if not os.path.exists(dst_image_path):
            os.mkdir(dst_image_path)
        if not os.path.exists(dst_xml_path):
            os.mkdir(dst_xml_path)

        raw_anno_list = list_all_files(anno_dir, exts=['txt'])
        #print(raw_anno_list[0])
        for idx,ra in enumerate(raw_anno_list):
            if idx % 100 == 0:
                print("at %s / %s"%(idx, len(raw_anno_list)))

            rawimg     = ".".join(ra.split(".")[:-2]) + ".jpg"
            temp       = os.path.splitext(os.path.basename(ra))[0]
            temp       = ".".join(temp.split(".")[:-1])
            image_path = dst_image_path + temp + '.jpg'
            xml_path   = dst_xml_path   + temp + '.xml'

            cmd = "cp %s %s"%(rawimg, image_path)
            shell.run_system_command(cmd)

            #print("rawimg:"     , rawimg)
            #print("image_path:" , image_path)
            #print("xml_path:"   , xml_path)

            with open(ra, 'r') as f:
                for line in f.readlines():
                    words  = line.strip().split(',')
                    xmin   = int(words[0])
                    ymin   = int(words[1])
                    width  = int(words[2])
                    height = int(words[3])
                    xmax   = xmin + width
                    ymax   = ymin + height
                    gen_pascal_xml( image_path   = image_path, 
                                    size         = [width, height, 3],
                                    box          = [xmin, ymin, xmax, ymax],
                                    xml_dst_path = xml_path)


if __name__ == '__main__':
    #dslist= ['ch00006_20190204']
    dslist= ['ch00006_20190206']
    #dslist= ['test_dir']
    gen_pascal_anno(txt_anno_dirlist=dslist)
