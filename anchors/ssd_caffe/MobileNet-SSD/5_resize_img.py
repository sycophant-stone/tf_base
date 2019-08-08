import cv2
import os
import numpy as np
import shell
# list all image files
def list_all_files(dir_name, exts = ['jpg', 'bmp', 'png', 'xml']):
    result = []
    for dir, sub_dirs, file_names in os.walk(dir_name):
        for file_name in file_names:
            if any(file_name.endswith(ext) for ext in exts):
                result.append(os.path.join(dir, file_name))
    return result

def img_shift(img_path):
    img = cv2.imread(img_path,0)
    rows,cols = img.shape
    
    M = np.float32([[1,0,10],[0,1,50]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    baname = os.path.basename(img_path)
    baname = os.path.dirname(img_path)+"/"+ os.path.splitext(baname)[0]+"_shift.jpg"
    cv2.imwrite(baname,dst)

def filter_and_copy_img(img_dir_path):
    testfilename = img_dir_path+"FID_DID_HEAD_CLEAN_0_patches_int/test.txt"
    print("testfilename:", testfilename)
    with open(testfilename, 'r') as f:
        for line in f.readlines():
            words=line.strip().split(' ')
            imgpth = "%s/%s"%(img_dir_path, words[0])
            cmdline = "cp %s example/all_imgs"%(imgpth)
            #print(cmdline)
            shell.run_system_command(cmdline)



def img_resize(img_path):
    img = cv2.imread(img_path,0)
    rows,cols = img.shape
    dst = cv2.resize(img,(72, 72), interpolation=cv2.INTER_CUBIC)
    baname = os.path.basename(img_path)
    #baname = os.path.dirname(img_path)+"/"+ os.path.splitext(baname)[0]+"_sz72.jpg"
    baname = "example/all_imgs_72/"+ os.path.splitext(baname)[0]+"_sz72.jpg"
    cv2.imwrite(baname,dst)
    print(baname)

   
   
if __name__ == '__main__':
    #filter_and_copy_img(img_dir_path = "/ssd/hnren/tf/1sd/caffe/data/5_patches300to128_INT/HeadVocFormat/")
    
    ori_imgs_folder ="example/all_imgs"
    imgs = list_all_files(ori_imgs_folder, exts=["jpg"])
    for i, img_path in enumerate(imgs):
        img_resize(img_path)
