#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=examples/imagenet
DATA=data/ilsvrc12
TOOLS=build/tools

../caffe/build/tools/compute_image_mean /ssd/hnren/tf/1sd/MobileNet-SSD/5_patches300to128_INT_patches_inte_trainval_lmdb \
  imagenet_mean.binaryproto

echo "Done."
