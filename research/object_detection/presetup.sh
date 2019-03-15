!/bin/bash
cd vvv
mkdir exp pretrain tflite trainout
cd pretrain
echo "download ssh pretrain model"
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

echo "unzip "
tar xzf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
mv ssd_mobilenet_v2_coco_2018_03_29.tar/* .
rm -rf ssd*
cd ..