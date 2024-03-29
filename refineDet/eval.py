from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils import tfrecord_voc_utils as voc_utils
import tensorflow as tf
import numpy as np
import RefineDet as net
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import test_vectorize_metrics as tvm
from skimage import io, transform
from utils.voc_classname_encoder import classname_to_ids
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
lr = 0.0001
batch_size = 1#32  # eval 
buffer_size = 1024
epochs = 300
reduce_lr_epoch = []
ckpt_path = os.path.join('.', './pretrained/vgg_16.ckpt')
config = {
    'mode': 'eval',                            # 'train' ,'test' , 'eval'
    'input_size': 320,                          # 320 for refinedet320, 512 for refinedet512
    'data_format': 'channels_last',             # 'channels_last' ,'channels_first'
    'num_classes': 20,
    'weight_decay': 1e-4,
    'keep_prob': 0.5,                           # not used
    'batch_size': batch_size,
    'nms_score_threshold': 0.1,
    'nms_max_boxes': 20,
    'nms_iou_threshold': 0.45,
    'pretraining_weight': ckpt_path
}

image_augmentor_config = {
    'data_format': 'channels_last',
    'output_shape': [320, 320],
    'zoom_size': [330, 330],
    'crop_method': 'random',
    'flip_prob': [0.0, 0.5],
    'fill_mode': 'BILINEAR',
    'keep_aspect_ratios': False,
    'constant_values': 0.,
    # 'rotate_range': None,
    'pad_truth_to': 60,
}

data1 = os.listdir('./eval_tfrecord')
data1 = [os.path.join('./eval_tfrecord', name) for name in data1]
data = []
for itm in data1:
    print("tim",itm)
    if "tfrecord" in itm:
        data.append(itm)

print("[test] data:%s"%(data))
eval_gen = voc_utils.get_generator(data,
                                    batch_size, buffer_size, image_augmentor_config)
print("[eval_gen]:",eval_gen)
evalset_provider = {
    'data_shape': [320, 320, 3],
    #'num_train': 5011,
    'num_val': 5011,                               # not used
    'eval_generator': eval_gen,  # not used in `test` mode
    #'val_generator': None                       # not used
}
refinedet = net.RefineDet320(config, evalset_provider)
refinedet.load_weight('refinedet320/test-2496')
#refinedet.load_weight('refinedet320/checkpoint')

# debug one-epoch
for i in range(1):#(epochs):
#for i in range(epochs):
    print('-'*25, 'epoch', i, '-'*25)
    if i in reduce_lr_epoch:
        lr = lr/10.
        print('reduce lr, lr=', lr, 'now')
    pred,gt = refinedet.eval_calc()
    #refinedet.calc_precision(pred,gt)
    '''
    print(pred[1])
    print("gt",gt)
    '''
    iou = tvm.calc_iou_vectorized(np.array(pred[1]),gt)
    '''
    print('>> pred', pred)
    print('>> gt', gt)
    '''
    #print('>> iou', iou)
    precision,tp,fp = tvm.calc_precision(iou,0.5)
    recall,_,fn = tvm.calc_recall(iou,0.5)
    print('>> p:%d,r:%d,tp:%d,fp:%d,fn:%d'%(precision,recall,tp,fp,fn))
    #refinedet.save_weight('latest', './refinedet320/test')    # 'latest' 'best'
refinedet.release_resorce()
# img = io.imread('000026.jpg')
# img = transform.resize(img, [300,300])
# img = np.expand_dims(img, 0)
# result = ssd300.test_one_image(img)
# id_to_clasname = {k:v for (v,k) in classname_to_ids.items()}
# scores = result[0]
# bbox = result[1]
# class_id = result[2]
# print(scores, bbox, class_id)
# plt.figure(1)
# plt.imshow(np.squeeze(img))
# axis = plt.gca()
# for i in range(len(scores)):
#     rect = patches.Rectangle((bbox[i][1],bbox[i][0]), bbox[i][3]-bbox[i][1],bbox[i][2]-bbox[i][0],linewidth=2,edgecolor='b',facecolor='none')
#     axis.add_patch(rect)
#     plt.text(bbox[i][1],bbox[i][0], id_to_clasname[class_id[i]]+str(' ')+str(scores[i]), color='red', fontsize=12)
# plt.show()
