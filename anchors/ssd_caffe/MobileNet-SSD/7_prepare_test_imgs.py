import numpy as np
import os
import cv2
import shell

# list all image files
def list_all_files(dir_name, exts = ['jpg', 'bmp', 'png', 'xml']):
    result = []
    for dir, sub_dirs, file_names in os.walk(dir_name):
        for file_name in file_names:
            if any(file_name.endswith(ext) for ext in exts):
                result.append(os.path.join(dir, file_name))
    return result

def filter_and_copy_img(img_dir_path):
    testfilename = img_dir_path+"test.txt"
    with open(testfilename, 'r') as f:
        for line in f.readlines():
            words=line.strip().split(' ')
            imgpth = "%s/%s"%(img_dir_path, words[0])
            cmdline = "cp %s example/all_imgs"
            run_system_command(cmdline)




if __name__ =='__main__':
    filter_and_copy_img(img_dir_path="/ssd/hnren/tf/1sd/caffe/data/5_patches300to128_INT/HeadVocFormat/")
