#!/usr/bin/python
# -*- coding: cp936 -*-
import os
import cv2
import numpy as np
def drawShape(pts,img):
    if len(pts)/2==42:
        comp = [[0,1,2,3,4,5,6,7], [8,9,10,11,12,13,14,15,8,16], [17,18,19,20,21,22,23,24,25],
                [26,27,28,29,30],[31,32,28,33,34], [35,36,37,38,39,40,41]]
    if len(pts)/2 ==72:
        comp = [[0,1,2,3,4,5,6,7,8,9,10,11,12],
		[13,14,15,16,17,18,19,20,13,21],
		[22,23,24,25,26,27,28,29,22],
                [30,31,32,33,34,35,36,37,30,38],
		[39,40,41,42,43,44,45,46,39],
		[47,48,49,50,51,52,53,54,55,56,47],
		[51,57,52],
		[58,59,60,61,62,63,64,65,58],
		[58,66,67,68,62,69,70,71,58]]
    else:
        comp = []
        
    for i in xrange(len(comp)):
        for j in xrange(len(comp[i])-1):
            cv2.line(img,(int(pts[2*comp[i][j]]),int(pts[2*comp[i][j]+1])),(int(pts[2*comp[i][j+1]]),int(pts[2*comp[i][j+1]+1])),(0,255,0),2)
    for j in xrange(len(pts)//2):
        cv2.circle(img,(int(pts[2*j]),int(pts[2*j+1])),2,(255,0,0),-1)
    #cv2.imshow("viewShape",img)
    #key = cv2.waitKey(0)
