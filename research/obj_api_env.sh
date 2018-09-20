#!/bin/bash
echo "export"
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
echo "protoc"
/home/julyedu_433249/work/tf_base/research/bin/protoc object_detection/protos/*.proto --python_out=.
