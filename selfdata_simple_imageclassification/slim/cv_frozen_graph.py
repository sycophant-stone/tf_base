from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.platform import app
from tensorflow.python.platform import gfile
from tensorflow.python.training import saver as saver_lib

graphfilepath="satellite/inception_v3_inf_graph.pb"
input_checkout_filepath="satellite/train_dir/model.ckpt-0"
output_node_names = "InceptionV3/Predictions/Reshape_1"
output_graph = "satellite/frozen_graph.pb"
def _parse_input_graph_proto(graphfilepath):
    '''
    从二进制的保存的graph恢复
    '''
    print("[_parse_input_graph_proto]: graphfilepath",graphfilepath)
    input_graph_def=graph_pb2.GraphDef()
    with gfile.FastGFile(graphfilepath,"rb") as f:
        input_graph_def.ParseFromString(f.read()) # 因为保存的是二进制.
    
    return input_graph_def


def frozen_this_graph():
    # 一, 恢复graph
    input_graph_def=_parse_input_graph_proto(graphfilepath)
    
    # 二, froze
    
    # 2.1 清理"显示"的device名字,便于移植
    for node in input_graph_def.node:
        node.device=""
    # 2.2 引入graph
    _ = importer.import_graph_def(input_graph_def,name="")
    
    with session.Session() as sess:
        var_list={}
        reader=pywrap_tensorflow.NewCheckpointReader(input_checkout_filepath)
        var_to_shapemap=reader.get_variable_to_shape_map()
        for key in var_to_shapemap:
            print("[frozen_this_graph]: var:",key)
            try:
                temp_tensor=sess.graph.get_tensor_by_name(key+':0')
            except KeyError:
                continue
            var_list[key]=temp_tensor
        # 2.3 把权值恢复到graph
        saver=saver_lib.Saver(var_list=var_list)
        saver.restore(sess,input_checkout_filepath)
        output_graph_def=graph_util.convert_variables_to_constants(sess,input_graph_def, output_node_names.split(","))
        
    # 保存输出
    if output_graph: 
        with gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())            
            
    print("%d ops in the final graph." % len(output_graph_def.node))

if __name__ == '__main__':
    frozen_this_graph()