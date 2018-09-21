for deeplab train.


1. 先运行,不然有些包找不到
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim


2. train时候的命令.注意 --tf_initial_checkpoint的用法

python deeplab/train.py     --logtostderr     --training_number_of_steps=30000     --train_split="train"     --model_variant="xception_65"     --atrous_rates=6     --atrous_rates=12     --atrous_rates=18     --output_stride=16     --decoder_output_stride=4     --train_crop_size=513     --train_crop_size=513     --train_batch_size=1     --dataset="pascal_voc_seg"     --tf_initial_checkpoint=deeplab/init_models/deeplabv3_pascal_train_aug/model.ckpt     --train_logdir=deeplab/trainlog     --dataset_dir=deeplab/datasets/pascal_voc_seg/tfrecord
