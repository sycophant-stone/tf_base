#!/bin/sh
#latest=snapshot/mobilenet_iter_73000.caffemodel
latest=$(ls -t snapshot/*.caffemodel | head -n 1)
if test -z $latest; then
	exit 1
fi
../caffe/build_debug/tools/caffe train -solver="solver_test_5_debug.prototxt" \
--weights="snapshot/mobilenet_iter_80000.caffemodel" \
-gpu 3
