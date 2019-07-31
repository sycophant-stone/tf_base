import os
import random

trainval_percent = 0.8    # train:test=8:2
train_percent = 0.7       # train / trainval = 0.7 
fdir = '/ssd/hnren/0_DATA/head_detection/fid_0/HeadVocFormat/FID_DID_HEAD_CLEAN_0_patches/ImageSets/Main/'      
xmlfilepath = '/ssd/hnren/0_DATA/head_detection/fid_0/HeadVocFormat/FID_DID_HEAD_CLEAN_0_patches/Annotations/'  
txtsavepath = fdir
total_xml = os.listdir(xmlfilepath)

num=len(total_xml)
list=range(num)
tv=int(num*trainval_percent)
tr=int(tv*train_percent)
trainval= random.sample(list,tv)
train=random.sample(trainval,tr)

ftrainval = open(fdir + 'trainval.txt', 'w')
ftest = open(fdir + 'test.txt', 'w')
ftrain = open(fdir + 'train.txt', 'w')
fval = open(fdir + 'val.txt', 'w')

for i  in list:
    name=total_xml[i][:-4]+'\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest .close()
