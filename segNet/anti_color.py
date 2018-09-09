

import cv2
import numpy as np

def anticolor():
    print("anticolor")
    pngimg=cv2.imread("0001TP_006690.png")
    pngimggray=cv2.cvtColor(pngimg,cv2.COLOR_BGR2GRAY)
    pngimggray=np.mat(pngimggray)
    pngimggray=255-pngimggray
    cv2.imshow("anticolor",pngimggray)
    k=cv2.waitKey(0)#无限期等待输入
    if k == 27:  # 如果输入ESC退出
        cv2.destroyAllWindows()

if __name__== '__main__':
    anticolor()
