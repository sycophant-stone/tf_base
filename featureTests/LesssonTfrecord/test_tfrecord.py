#%matplotlib inline
import numpy as np
#import skimage.io as io
from skimage import io
import cv2
img=io.imread('D:\\work\\stuff\\modules\\misc\\sprd_camera\\alg\\july\\tf_base\\featureTests\\LesssonTfrecord\\miniVOC\\JPEGImages\\2007_000175.jpg')
io.imshow(img)
cv2.waitKey (0)
cv2.destroyAllWindows()
