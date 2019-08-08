#!/bin/sh
#latest=snapshot/mobilenet_iter_73000.caffemodel
../caffe/build/tools/caffe train -solver="solver_test_5.prototxt" \
--weights="snapshot/mobilenet_iter_104000.caffemodel" \
-gpu 3
