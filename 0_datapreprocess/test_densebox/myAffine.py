#!/usr/bin/python
import os
import string
import cv2
import random
import math
import numpy as np
def Get_Affine_matrix(srcCenter, dstCenter, alpha,  scale):
    M = np.zeros((2,3), np.float32)
    M[0,0] = scale * math.cos(alpha)
    M[0,1] = scale * math.sin(alpha)
    M[1,0] = -M[0,1]
    M[1,1] =  M[0,0]
    M[0,2] = srcCenter[0] - M[0,0] * dstCenter[0] - M[0,1] * dstCenter[1]
    M[1,2] = srcCenter[1] - M[1,0] * dstCenter[0] - M[1,1] * dstCenter[1]
    return M
def inverseMatrix(M):
    D = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    D = 1.0/D if D != 0 else 0
    inv_M = np.zeros((2,3), np.float32)
    inv_M[0,0] = M[1,1]*D
    inv_M[0,1] = M[0,1]*(-D)
    inv_M[1,0] = M[1,0]*(-D)
    inv_M[1,1] = M[0,0]*D
    inv_M[0,2] = -inv_M[0,0]*M[0,2] - inv_M[0,1]*M[1,2]
    inv_M[1,2] = -inv_M[1,0]*M[0,2] - inv_M[1,1]*M[1,2]
    return inv_M
def Affine_Point(M,srcPt,dstPt):
    dstPt[0] = (M[0,0]*srcPt[0] + M[0,1]*srcPt[1] + M[0,2])
    dstPt[1] = (M[1,0]*srcPt[0] + M[1,1]*srcPt[1] + M[1,2])
    
