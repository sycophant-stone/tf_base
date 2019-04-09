#coding:utf-8
import sys
import numpy as np
import cv2
import os

rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.insert(0, rootPath)


def gen_pos_haar_like(srcDataSet, srcAnnotations):
    srcDataSet = os.path.join(rootPath, srcDataSet)
    srcAnnotations = os.path.join(rootPath, srcAnnotations)
    saveFolder = os.path.join(rootPath, "haar_like_tmp/")
    print(">>>>>> Gen positive sample for haar like...")
    typeName = ["pos", "neg", "part"]
    saveFiles = {}
    for tp in typeName:
        _saveFolder = os.path.join(saveFolder, tp)
        if not os.path.isdir(_saveFolder):
            os.makedirs(_saveFolder)
        saveFiles[tp] = open(os.path.join(saveFolder, "{}.txt".format(tp)), 'w')

    annotationsFile = open(srcAnnotations, "r")

    idx = 0
    for annotation in annotationsFile:
        annotation = annotation.strip().split(' ')
        # image path
        imPath = annotation[0]
        # boxed change to float type
        bbox = map(float, annotation[1:])
        # gt. each row mean bounding box
        x1,y1,x2,y2=annotation[1],annotation[2],annotation[3],annotation[4]
        idx += 1
        save_file = os.path.join(imPath)
        saveFiles['pos'].write(save_file + ' 1 %s %s %s %s\n'%(x1, y1, x2, y2))

    for f in saveFiles.values():
        f.close()
if __name__ == "__main__":
    gen_pos_haar_like("dataset/lfw_5590/", "dataset/trainImageList_ori.txt")
