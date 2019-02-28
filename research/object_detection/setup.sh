!/bin/sh
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

echo "install lxml"
pip install lxml
echo "create tfRecord"
python dataset_tools/create_pascal_tf_record.py --data_dir vvv/VOCdevkit/ --year=VOC2012 --set=train --output_path=vvv/pascal_train.record
python dataset_tools/create_pascal_tf_record.py --data_dir vvv/VOCdevkit/ --year=VOC2012 --set=val --output_path=vvv/pascal_val.record
echo "copy pbtxt"
cp data/pascal_label_map.pbtxt vvv
git config --global user.email "kawayi_rendroid@163.com"
git config --global user.name "kawayi"
