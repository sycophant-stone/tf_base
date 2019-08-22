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

def filter_and_copy_img(img_dir_path, dst_dir):
    testfilename = img_dir_path+"FID_DID_HEAD_CLEAN_0_patches_int/test.txt"
    with open(testfilename, 'r') as f:
        for line in f.readlines():
            words=line.strip().split(' ')
            imgpth = "%s/%s"%(img_dir_path, words[0])
            cmdline = "cp %s %s"%(imgpth, dst_dir)
            print(cmdline)
            shell.run_system_command(cmdline)




if __name__ =='__main__':
    DST_DIR = "example/all_imgs_in72"
    if not os.path.exists(DST_DIR):
        os.mkdir(DST_DIR)
    filter_and_copy_img(img_dir_path="/ssd/hnren/tf/1sd/caffe/data/5_patches300to72_INT/HeadVocFormat/", dst_dir = DST_DIR)
