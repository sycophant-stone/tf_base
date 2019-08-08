#!/bin/sh
../caffe/build/tools/caffe time --model="example_5_patches300to128_INT_patches_inte/MobileNetSSD_train.prototxt" \
--weights="snapshot/86/mobilenet_iter_105000.caffemodel" \
-gpu 3
