root_dir="/ssd/hnren/tf/tf_base/refineDet_caffe"
test_metrics_path="test/unds_metrics"
current_workpath="$root_dir/$test_metrics_path"
if [ -f "$root_dir/$test_metrics_path/voc2007_img.txt" ]
then
  rm -f "$root_dir/$test_metrics_path/voc2007_img.txt"
fi
cp $root_dir/data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt  $current_workpath/voc2007_img.txt 
sed -i "s/^/VOC2007\/JPEGImages\//g" $current_workpath/voc2007_img.txt
