
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
#from matplotlib import pyplot as plt
#from PIL import Image

from utils import label_map_util

from utils import visualization_utils as vis_util


PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def test_single_picture():
	# 1. load frozen tensorflow model into memory
	detection_graph=tf.Graph()
	with detection_graph.as_default():
		od_graph_def=tf.GraphDef()
		with tf.gfile.GFile("voc/export/frozen_inference_graph.pb",'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def,name='')
	
	# 2. load label map,which contains map indices to category names.
	label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
	categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)
	
	# 3. detection
	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
			# 3.1 Define Input and Output
			image_tensor=detection_graph.get_tensor_by_name("image_tensor:0")
			
			# 3.2 detection box
			detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			
			# 3.3 scores
			detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
			detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')
			
			for image_path in TEST_IMAGE_PATHS:
				image=Image.open(image_path)
				image_np = load_image_into_numpy_array(image)
				image_np_expanded = np.expand_dims(image_np, axis=0)
				(boxes, scores, classes, num) = sess.run(
				[detection_boxes, detection_scores, detection_classes, num_detections],
				feed_dict={image_tensor: image_np_expanded})
				
				print("detection result: boxes:",boxes)
				print("detection result: scores:",scores)
				print("detection result: classes:",classes)
				print("detection result: num:",num)
			


if __name__=="__main__":
	test_single_picture()