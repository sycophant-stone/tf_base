#!/bin/sh
if ! test -f example/MobileNetSSD_train.prototxt ;then
	echo "error: example/MobileNetSSD_train.prototxt does not exist."
	echo "please use the gen_model.sh to generate your own model."
        exit 1
fi
mkdir -p snapshot
../caffe/build/tools/caffe train -solver="solver_5_patches300to128_INT_patches_inte.prototxt" \
--snapshot="snapshot/mobilenet_iter_79215.solverstate" \
-gpu 4,5,6,7 
