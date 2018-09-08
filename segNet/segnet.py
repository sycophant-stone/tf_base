'''
   This program is for SegNet.
'''

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import os, sys
import numpy as np
import math
from datetime import datetime
import time
from PIL import Image
from math import ceil
from tensorflow.python.ops import gen_nn_ops
# modules
#from Utils import _variable_with_weight_decay, _variable_on_cpu, _add_loss_summaries, _activation_summary, print_hist_summery, get_hist, per_class_acc, writeImage
#from Inputs import *

'''------------------------------------------------------------------'''
##input

def get_filename_list(filename):
	# 拿到数据集
	fd =open(filename)
	for line in fd:
		line=line.strip().split(" ")
		image_path=line[0]
		label_path=line[1]
	
	return image_path,label_path



'''------------------------------------------------------------------'''

def training(trainfilepath,valfilepath):
	train_image_filenames,train_label_filenames=get_filename_list(trainfilepath)
	val_image_filenames,val_label_filenames=get_filename_list(valfilepath)
	
    with tf.Graph




		
