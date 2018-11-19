#%matplotlib inline
import numpy as np
#import skimage.io as io
from skimage import io
import cv2
#filepath="D:\\work\\stuff\\modules\\misc\\sprd_camera\\alg\\july\\tf_base\\featureTests\\LesssonTfrecord\\miniVOC\\JPEGImages\\2007_000175.jpg"
filepath="./miniVOC/JPEGImages/2007_000175.jpg"
img=io.imread(filepath)
io.imshow(img)
