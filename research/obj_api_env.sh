#!/bin/bash
echo "export"
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
echo "protoc"
/work/bin/protoc object_detection/protos/*.proto --python_out=.
