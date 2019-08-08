import cv2
import os
import numpy as np

def img_shift(img_path):
    img = cv2.imread(img_path,0)
    rows,cols = img.shape
    
    M = np.float32([[1,0,10],[0,1,50]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    baname = os.path.basename(img_path)
    baname = os.path.dirname(img_path)+"/"+ os.path.splitext(baname)[0]+"_shift.jpg"
    cv2.imwrite(baname,dst)
   
if __name__ == '__main__':
    img_shift(img_path="example/images_fid01/ch00006_20190206_ch00006_20190206142936.mp4.cut.mp4_000000_crop_0.jpg")
