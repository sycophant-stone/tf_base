#cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
cur_dir="/ssd/hnren/tf/tf_base/refineDet_caffe/test/unds_metrics/"
#root_dir=$cur_dir/../..
root_dir="/ssd/hnren/tf/tf_base/refineDet_caffe/test/unds_metrics/"

#cd $root_dir
cd /ssd/hnren/tf/tf_base/refineDet_caffe/test/unds_metrics

redo=1
data_root_dir="/ssd/hnren/tf/tf_base/refineDet_caffe/test/unds_metrics/VOCdevkit/"
dataset_name="VOC0712"
mapfile="/ssd/hnren/tf/tf_base/refineDet_caffe/data/VOC0712/labelmap_voc.prototxt"
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded --redo"

for subset in trainval test
do
  echo " python $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir /ssd/hnren/tf/tf_base/refineDet_caffe/data/VOC0712/$subset.txt $data_root_dir/$dataset_name/$db/$dataset_name"_"$subset"_"$db examples/$dataset_name"
  python $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir /ssd/hnren/tf/tf_base/refineDet_caffe/data/VOC0712/$subset.txt $data_root_dir/$dataset_name/$db/$dataset_name"_"$subset"_"$db examples/$dataset_name
done
