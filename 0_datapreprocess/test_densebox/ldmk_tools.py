import os, sys
import numpy as np
import math
from myAffine import *
from drawShape import *

def distance(pt1, pt2):
    return math.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)

def decide_face_type(pts):
    # 0: frontal face
    # 1: left profile
    # 2: right profile
    face_type = 0
    if (abs(pts[13*2]+1)<0.01 and abs(pts[13*2+1]+1)<0.01):
        face_type = 1  # left profile faces
    elif (abs(pts[34*2]+1)<0.01 and abs(pts[34*2+1]+1)<0.01):
        face_type = 2  # right profile
    else:
        eye1 =[pts[13*2],pts[13*2+1]]
        eye2 =[pts[34*2],pts[34*2+1]]
        p1 = [pts[0],pts[1]]  #left profile
        p2 = [pts[12*2],pts[12*2+1]]
        face_size = distance(p1, p2)
        eye_dist = distance(eye1, eye2)
        dist1 = distance(p1, eye1)
        dist2 = distance(p2, eye2)
        if eye_dist >= 0.6 * face_size:
            face_type = 0
        elif dist1 < dist2:
            face_type = 1
        else:
            face_type = 2

    return face_type

def getBox(pts):
    face_type = decide_face_type(pts)
    pt1 =[(pts[47*2] + pts[56*2]) / 2, (pts[47*2+1] + pts[56*2 + 1]) / 2 ] # nose top
    pt2 =[pts[6*2],pts[6*2+1]] #chin
    alpha = math.atan2(pt1[1]-pt2[1],pt1[0]-pt2[0]) + (3.14159 / 2)

    #affine warping the face
    srcCenter = np.array([(pt1[0]+pt2[0])/2.0,(pt1[1]+pt2[1])/2.0], np.float32)
    dstCenter = np.array([200,200],np.float32)
    
    scale = 1
    warp_mat = Get_Affine_matrix(dstCenter,srcCenter,alpha,scale)
    
    min_x = 100000
    min_y = 100000
    max_x = -100000
    max_y = -100000
    for n in range(len(pts)/2):
        srcpt = np.array([pts[2*n],pts[2*n+1]], np.float32)
        dstpt = np.array([0,0], np.float32)
        if srcpt[0]!=-1 and srcpt[1]!=-1:
            Affine_Point(warp_mat,srcpt,dstpt)
            if min_x > dstpt[0]:
                min_x = dstpt[0]
            if min_y > dstpt[1]:
                min_y = dstpt[1]
            if max_x < dstpt[0]:
                max_x = dstpt[0]
            if max_y < dstpt[1]:
                max_y = dstpt[1]
        else:
            dstpt[0] = -1
            dstpt[1] = -1

    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    fw = max_x - min_x + 1
    fh = max_y - min_y + 1
    fsize = max(fh, fw)

    # Transform (cx, cy) back to the original image
    inv_warp_mat = inverseMatrix(warp_mat)
    srcpt = np.array([cx, cy], np.float32)
    dstpt = np.array([0,0], np.float32)
    Affine_Point(inv_warp_mat, srcpt, dstpt)
    cx = dstpt[0]
    cy = dstpt[1]

    return (cx, cy, fsize, alpha, face_type)

def load_face_boxes(line):
    boxes = []
    if "," in line:
        words = line.strip().split(",")
    else:
        words = line.strip().split(" ")
    
    if len(words) < 145:
        return (None, None)

    imgName = words[0]
    pts = map(float, words[1:])
    nface = len(pts) / 144
    for i in range(nface):
        boxes.append(getBox(pts[144*i : 144*(i+1)]))
    return (imgName, boxes)
    
