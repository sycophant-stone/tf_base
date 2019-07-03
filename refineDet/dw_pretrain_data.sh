echo "download pretrained Vgg"
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
echo "copy VOC2007"
cp /data_public/VOC/voc2007/VOCtrainval_06-Nov-2007.tar .
echo "restore last checkpoint"
mkdir refinedet320
cp /data/refinedet_trainout/* refinedet320/
