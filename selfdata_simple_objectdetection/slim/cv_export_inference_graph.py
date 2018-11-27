from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.platform import gfile
from datasets import dataset_factory
from nets import nets_factory

def export_graph_case():
    with tf.Graph().as_default() as graph:
        # 第一步,拿到dataset描述符.
        dataset=dataset_factory.get_dataset(name="satellite",split_name='validation',
                                    dataset_dir="satellite/data")
        # 第二步,拿到网络结构
        network_fn = nets_factory.get_network_fn("inception_v3",
                                   num_classes=dataset.num_classes,
                                   is_training=False)
        if hasattr(network_fn,"default_image_size"):
            image_size=network_fn.default_image_size
        else:
            image_size=224 # default value
        # 第三步,该网路没有入口节点,创建input节点,并和inception网络关联.
        input_placeholder=tf.placeholder(name='input',
                                        dtype=tf.float32,
                                        shape=[1,image_size,image_size,3])
        network_fn(input_placeholder) #把这个input节点和inception网络挂钩.
        
        # 第四步,获取当前的网络图.
        graphdef=graph.as_graph_def()
        with gfile.GFile("satellite/inception_v3_inf_graph.pb","wb") as f:
            f.write(graphdef.SerializeToString())

if __name__ == '__main__':
    export_graph_case()