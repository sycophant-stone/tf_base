from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

def preprecess(image,height,width,central_crop_factor=0.875,scope=None):
    print("[preprocess]: image:",image)
    print("[preprocess]: height:",height)
    print("[preprocess]: width:",width)
    print("[preprocess]: central_crop_factor:",central_crop_factor)
    with tf.name_scope(scope,"eval_name",[image,height,width]):
        if image.dtype!=tf.float32:
            image=tf.image.convert_image_dtype(image,dtype=tf.float32)
        if central_crop_factor:
            print("[preprocess]: before central crop:",image)
            image=tf.image.central_crop(image,central_fraction=central_crop_factor)
            print("[preprocess]: after central crop:",image)
        if height and width:
            print("[preprocess]: before image:",image)
            image=tf.expand_dims(image,0)
            print("[preprocess]: after expand_dims:",image)
            image=tf.image.resize_bilinear(image,[height,width],align_corners=True)
            print("[preprocess]: after resize :",image)
            image=tf.squeeze(image,[0])
            print("[preprocess]: after image sequeeze on [0]: ",image)
        image=tf.subtract(image,0.5)
        image=tf.multiply(image,2.0)
        return image
            
            
class NodeLookup(object):
  def __init__(self, label_lookup_path=None):
    self.node_lookup = self.load(label_lookup_path)

  def load(self, label_lookup_path):
    node_id_to_name = {}
    with open(label_lookup_path) as f:
      for index, line in enumerate(f):
        node_id_to_name[index] = line.strip()
    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]        

def test_obd_one_image():
    # create input data
    imagefile="test.jpg"
    with tf.Graph().as_default():
        image_data=tf.gfile.FastGFile(imagefile,"rb").read()
        image_data=tf.image.decode_jpeg(image_data)
        image_data=preprecess(image_data,299,299)
        image_data=tf.expand_dims(image_data,0)
        with tf.Session() as sess:
            image_data = sess.run(image_data)
        
        #create_graph()
    with tf.gfile.FastGFile("satellite/inception_v3_inf_graph.pb",'rb') as f:
        graphdef=tf.GraphDef()
        graphdef.ParseFromString(f.read())
        tf.import_graph_def(graphdef,name='')
    
    with tf.Session() as sess:
        softmax_tensor=sess.graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')
        predictions=sess.run(softmax_tensor,{'input:0':image_data}) #给input:0传入image_data
        print("[test_obd_one_image]: predictions",predictions)
        predictions=np.sequeeze(predictions)
        print("[test_obd_one_image]: predictions after sequeeze",predictions)
        nodelookup=NodeLookup("satellite/data/label.txt")
        
        top_k=predictions.argsort()[-5:][::-1]
        for nodeid in top_k:
            human_reading=nodelookup.id_to_string(node_id)
            score=predictions[node_id]
            print("%s (score = %.5f)" %(human_reading, score))

if __name__ == '__main__':
    test_obd_one_image()