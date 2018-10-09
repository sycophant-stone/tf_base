import tensorflow as tf
slim=tf.contrib.slim
prefetch_queue = slim.prefetch_queue
import collections
from tensorflow.contrib.slim.nets import resnet_utils
from deployment import model_deploy
LOGITS_SCOPE_NAME = 'logits'
MERGED_LOGITS_SCOPE = 'merged_logits'
IMAGE_POOLING_SCOPE = 'image_pooling'
ASPP_SCOPE = 'aspp'
CONCAT_PROJECTION_SCOPE = 'concat_projection'
DECODER_SCOPE = 'decoder'

#-----------------------xception65网络----------------------
def stack_blocks_dense(net,
                       blocks,
                       output_stride=None,
                       outputs_collections=None):
    """实现block描述的net
       其中unit_fn是xception_module函数
    """
    current_stride=1
    rate=1
    for block in blocks:
        with tf.variable_scope(block.scope,'block',[net]) as sc:
            for i,unit in enumerate(block.args):
                with tf.variable_scope('unit_%d'%(i+1),values=[net]):
                    if output_stride is not None and current_stride==output_stride:
                        net=block.unit_fn(net,rate=rate,**dict(unit,stride=1))
                        rate*=unit.get('stride',1)
                    else:
                        net=block.unit_fn(net,rate=1,**unit)
                        current_stride=unit.get('stride',1)
                    net=slim.utils.collect_named_outputs(outputs_collections,sc.name,net)
    
    return net
                    
    

def xception(inputs,
             blocks,
             num_classes=None,
             is_training=True,
             global_pool=True,
             keep_prob=0.5,
             output_stride=None,
             reuse=None,
             scope=None):
    """把Block描述的网络组织起来
    args:
        blocks: 描述了一系列的xception 网络block.
    """
    with tf.variable_scope(scope,'xception',
                          [inputs],reuse=reuse) as sc:
        end_points_collection=sc.original_name_scope+'end_points'
        with slim.arg_scope([slim.conv2d,
                            slim.separable_conv2d,
                            xception_module,
                            stack_blocks_dense],
                           outputs_collections=end_points_collection):
            with slim.arg_scope([slim.batch_norm],is_training=is_training):
                net=inputs
                if output_stride is not None:
                    if output_stride%2!=0:
                        raise ValueError('output_stride should be a mulitple of 2')
                output_stride/=2
                # xception的entry flow前面还有两个conv
                net=resnet_utils.conv2d_same(net,32,3,stride=2,scope='entry_flow/conv1_1')
                net=resnet_utils.conv2d_same(net,64,3,stride=1,scope='entry_flow/conv1_2')
                
                # 抽取blocks描述的网路
                net=stack_blocks_dense(net,blocks,output_stride)
                end_points=slim.utils.convert_collection_to_dict(end_points_collection,clear_collection=True)
                
                if global_pool:
                    net=tf.reduce_mean(net,[1,2],name='global_pool',keepdim=True)
                    end_points['global_pool']=net
                if num_classes:
                    net=slim.dropout(net,keep_prob=keep_prob,is_training=is_training,scope='prelogits_dropout')
                    net=slim.conv2d(net,num_classes=num_classes,[1,1],activation_fn=None,
                                   normalizer_fn=None,scope='logits')
                    end_points[sc.name+'/logits']=net
                    end_points['predictions']=slim.softmax(net,scope='predictions')
                return net,end_points

def separable_conv2d_same(inputs,
                          num_outputs,
                          kernel_size,
                          depth_multiplier,
                          stride,
                          rate=1,
                          use_explicit_padding=True,
                          regularize_depthwise=False,
                          scope=None,
                          **kwargs):
    """3x3的卷积,可分离卷积
    """
    # 两个辅助函数
    def _seperable_conv2d(padding):
        return slim.separable_conv2d(inputs,
                                     num_outputs,
                                     kernel_size,
                                     depth_muliplier=depth_multiplier,
                                     stride=stride,
                                     rate=rate,
                                     padding=padding,
                                     scope=scope,
                                     **kwargs)
    
    def _split_separable_conv2d(padding):
        # 这个里边输出节点没有是num_outputs
        outputs=slim.separable_conv2d(inputs,
                                     None,
                                     kernel_size,
                                     depth_multiplier=depth_multiplier,
                                     stride=stride,
                                     rate=rate,
                                     padding=padding,
                                     scope=scope+'_depthwise',
                                     **kwargs)
        # 然后加一个1x1的小卷积做的num_outputs.
        return slim.conv2d(outputs,
                          num_outputs,
                          1,
                          scope=scope+'_pointwise',
                          **kwargs)
    is_stride ==1 or not use_explicit_padding:
        if regularize_depthwise:
            # 加正则化,并不是downsampling,
            outputs=_seperable_conv2d(padding='SAME')
        else:
            outputs=_split_separable_conv2d(padding='SAME')
    else:
        if regularize_depthwise:
            outputs=_seperable_conv2d(padding='VALID')
        else:
            outputs=_split_separable_conv2d(padding='VALID')
            
    return outputs

def xception_module(inputs,
                    depth_list,
                    skip_connection_type,
                    stride,
                    unit_rate_list=None,
                    rate=1,
                    activation_fn_in_separable_conv=False,
                    regularize_depthwise=False,
                    outputs_collections=None,
                    scope=None):
    """ xception模块包括:
        'residual'和'shortcut'
        residual含有separable conv 3x3
        shortcut含有1x1 conv or not
        xception不使用max pooling.而是采用separable conv with striding.原因是当前max pooling不支持带洞操作.
        skip_connection_type: shortcut和residual的concat方法,conv,sum,none, conv是shortcut通路经过1x1小卷积和residual加和,
                              sum是residual和shortcut加和.
                              none只采用residual.
    """
    with tf.variable_scope(scope,'xception_module',[inputs]) as sc:
        residual=inputs
        
        # 功能函数,处理relu在sperable conv前还是后.
        def _separable_conv(features,depth,kernel_sise,depth_multiplier,
                            regularize_depthwise,rate,stride,scope):
            if activation_fn_in_separable_conv:
                activation_fn=tf.nn.relu
            else:
                activation_fn=None
                features=tf.nn.relu(features)
            return separable_conv2d_same(features,
                                   depth,
                                   kernel_size,
                                   depth_multiplier=depth_multiplier,
                                   stride=stride,
                                   rate=rate,
                                   activation_fn=activation_fn,
                                   regularize_depthwise=regularize_depthwise,
                                   scope=scope)
        for i in range(3):
            residual=_separable_conv(residual,depth_list[i],
                                    kernel_size=3,
                                    depth_multiplier=1,
                                    regularize_depthwise=regularize_depthwise,
                                    rate=rate*unit_rate_list[i],
                                    stride=stride if i==2 else 1,
                                    scope='separable_conv'+str(i+1))
        
        if skip_connection_type =='conv':
            shortcut=slim.conv2d(inputs,
                                depth_list[-1],
                                [1,1],
                                stride=stride,
                                activation_fn=None,
                                scope='shortcut')
            outputs=residual+shortcut
        elif skip_connection_type=='sum':
            outputs=residual+shortcut
        else: # None, 表示没有shortcut这个捷径
            outputs=residual
        
        return slim.utils.collect_named_outputs(outputs_collections,
                                               sc.name,
                                               outputs)
    

class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """xception模块 
    unit_fn: xception模块
    """
    

def xception_block(scope,
                   depth_list,
                   skip_connection_type,
                   activation_fn_in_separable_conv,
                   regularize_depthwise,
                   num_units,
                   stride,
                   unit_rate_list=None):
    """构建xception块.
    num_units: 描述相同的该块有多少个.
    
    """
    return Block(scope, xception_module, [{
      'depth_list': depth_list,
      'skip_connection_type': skip_connection_type,
      'activation_fn_in_separable_conv': activation_fn_in_separable_conv,
      'regularize_depthwise': regularize_depthwise,
      'stride': stride,
      'unit_rate_list': unit_rate_list,
  }] * num_units)
    
def xception_65(inputs,
                num_classes=None,
                is_training=True,
                global_pool=True,
                keep_prob=0.5,
                output_stride=None,
                regularize_depthwise=False,
                multi_grid=None,
                reuse=None,
                scope='xception_65'):
    """搭建Xception-65 模型
    """
    blocks=[
        xception_block('entry_flow/block1',
                       depth_list=[128,128,128],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('entry_flow/block2',
                       depth_list=[256,256,256],
                       skip_connection_type='conv',
                       activation_fn_in_separable_conv=False,
                       regularize_depthwise=regularize_depthwise,
                       num_units=1,
                       stride=2),
        xception_block('entry_flow/block3',
                      depth_list=[728,728,728],
                      skip_connection_type='conv',
                      activation_fn_in_separable_conv=False,
                      regularize_depthwise=regularize_depthwise,
                      num_units=1,
                      stride=2),
        
        xception_block('middle_flow/block1',
                      depth_list=[728,728,728],
                      skip_connection_type='sum',
                      activation_fn_in_separable_conv=False,
                      regularize_depthwise=regularize_depthwise,
                      num_units=16,
                      stride=1),
        
        xception_block('exit_flow/block1',
                      depth_list=[728,1024,1024],
                      skip_connection_type='conv',
                      activation_fn_in_separable_conv=False,
                      regularize_depthwise=regularize_depthwise,
                      num_units=1,
                      stride=2),
        xception_block('exit_flow/block2',
                      depth_list=[1536,1536,2048],
                      skip_connection_type='none',
                      activation_fn_in_separable_conv=True,
                      regularize_depthwise=regularize_depthwise,
                      num_units=1,
                      stride=1,
                      unit_rate_list=multi_grid),
        
    ]
    return xception(inputs,
                  blocks=blocks,
                  num_classes=num_classes,
                  is_training=is_training,
                  global_pool=global_pool,
                  keep_prob=keep_prob,
                  output_stride=output_stride,
                  reuse=reuse,
                  scope=scope)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#---------------------------------------------

def cal_scaled_dim_val(dim,scale_coeff):
    """利用scale_coeff对dim维做成scale的维度.
       这里只是一个计算,计算scale之后的维度数.其实并没有实际scale数据Tensor对象.
    """
    if isinstance(dim,tf.Tensor):
        return tf.cast((tf.tofloat(dim)-1.0)*scale_coeff+1.0,tf.int32) # 其实这里边的这个+1.0是为了向上取整
    else:
        return (float(dim)-1.0)*scale_coeff+1.0


def xception_arg_scope(weight_decay=0.00004,
                       batch_norm_decay=0.9997,
                       batch_norm_epsilon=0.001,
                       batch_norm_scale=True,
                       weights_initializer_stddev=0.09,
                       activation_fn=tf.nn.relu,
                       regularize_depthwise=False,
                       use_batch_norm=True):
    """生成xception65 使用的arg_scope.
    
    """
    batch_norm_params={
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
    }
    if regularize_depthwise:
        depthwise_regularizer=slim.l2_regularizer(weight_decay)
    else:
        depthwise_regularizer=None
    with slim.arg_scope(
    [slim.conv2d,slim.separable_conv2d],
        weights_initializer=tf.truncated_normal_initializer(stddev=weights_initializer_stddev),
        activation_fn=activation_fn,
        normalizer_fn=slim.batch_norm if use_batch_norm else None):
        with slim.arg_scope([slim.batch_norm],**batch_norm_params):
            with slim.arg_scope([slim.conv2d],
                               weights_regularizer=slim.l2_regularizer(weight_decay)):
                with slim.arg_scope([slim.separable_conv2d],weights_regularizer=depthwise_regularizer) as arg_sc:
                    
    return arg_sc

def _preprocess_zero_mean_unit_range(inputs):
    """把图像数据转成-1到1
    """
    return tf.to_float(inputs)/255*2.0 -1.0

def get_network(network_name, preprocess_images, arg_scope=None):
    """get network的函数以及对应的参数
    """
    arg_scope=arg_scope or xception_arg_scope() # 在xception默认参数配置基础上
    if preprocess_images==True:
        #preprocess_function =_preprocess_zero_mean_unit_range
    def network_fn(inputs, *args, **kwargs):
        with slim.arg_scope(arg_scope):
            return xception_65(_preprocess_zero_mean_unit_range(inputs),
                              *args,**kwargs)
    return network_fn
    

def local_extract_features(
    features,
    model_options,
    weight_decay=0.0001,
    reuse=None,
    is_training=False,
    fine_tune_batch_norm=False):
    """对于特定的模型抽取features
    
    """
    #做xception的model_variant.
    arg_scope=xception_arg_scope(weight_decay=weight_decay,batch_norm_decay=0.9997,
                               batch_norm_epsilon=1e-3,
                               batch_norm_scale=True,
                               regularize_depthwise=False)
    temp_network=get_network("exception",preprocess_images=True,arg_scope)
    features,endpoints=temp_network(inputs=features,
                 num_classes=None,
                is_training=is_training,
                global_pool=False,
                 output_stride=8,
                 multi_grid=None,
                 reuse=reuse,
                 scope='xception_65')
    return features,endpoints
    
    
def extract_features(features,
                     model_options,
                     weight_decay=0.0001,
                     reuse=None,
                     is_training=False,
                     fine_tune_batch_norm=False):
    """提取特征图和end_point.
       返回值:
       1. concat_logits, 它是一系列的融合.
       首先,主干输出的feature maps.经过image pool产生一个结果; 经过1x1的conv产生一个结果; 经过rates(6,12,18)的aspp产生一组结果.
       然后,这些结果会concat成一个输出.
       end_points
    """
    # 提取
    features,end_points=local_extract_features(
        images,
        output_stride=model_options.output_stride,
        multi_grid=model_options.multi_grid,
        model_variant=model_options.model_variant,
        depth_multiplier=model_options.depth_multiplier,
        weight_decay=weight_decay,
        reuse=reuse,
        is_training=in_training,
        fine_tune_batch_norm=fine_tune_batch_norm)
    
    if not model_options.aspp_with_batch_norm:
        # aspp不需要batch norm,直接返回features
        # 我们知道batch norm是在激活函数之前,对features做的.让其归一化到0~1之间.
        return features,end_points
    else:
        batch_norm_params={
            'is_training':is_training and fine_tune_batch_norm,
            'decay':0.9997,
            'spsilon':1e-5,
            'scale':True,
        }
    # slim.arg_scope对给定的op存储其param
    # 构建figure5里边的 Block4之后处理的ASPP部分.
    with slim.arg_scope(
        [slim.cov2d,slim.separable_conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm,
        padding='SAME',
        stride=1,
        reuse=reuse):
        # 目的是要做batch norm
        with slim.arg_scope(
            [slim.batch_norm],
            **batch_norm_params):
            depth=256
            branch_logits=[]
            # step 1 做一个pooling
            # 融合B部分(image pooling).
            if model_options.add_image_level_feature:
                
                if model_options.crop_size is not None:
                    image_pooling_crop_size=model_options.image_pooling_crop_size
                    if image_pooling_crop_size is None:
                        image_pooling_crop_size=model_options.crop_size
                    # 计算pooling的scale
                    pooling_height=cal_scaled_dim_val(image_pooling_crop_size[0],
                                                      1.0/model_options.output_stride)
                    pooling_width=cal_scaled_dim_val(image_pooling_crop_size[1],
                                                    1.0/model_options.output_stride)
                    # 加pooling层
                    image_feature_pooled=slim.avg_pool2d(features,
                                    [pooling_height,pooling_width],# 这个应该是kernel的size
                                    [1,1],# 这个应该是hw的strides
                                    padding='VALID')
                    # crop size 也需要做一下scale
                    resize_height=cal_scaled_dim_val(model_options.crop_size[0],
                                                    1.0/model_options.output_stride)
                    resize_width=cal_scaled_dim_val(model_ooptions.crop_size[1],
                                                   1.0/model_options.output_stride)
                else:
                    # 没有crop的size,做一个global的pooling
                    pooling_height=tf.shape(features)[0]
                    pooling_width=tf.shape(features)[1]
                    image_feature_pooled=tf.reduce_mean(
                        features,
                        axis=[1,2])[:,tf.newaxis,tf.newaxis] # 在features基础上再添加两个维度,但是这两个维度还没有其他的填充值.
                    resize_height=pooling_height
                    resize_width=pooling_width
                # 添加一个1x1的卷积
                image_feature=slim.conv2d(
                    image_feature_pooled,depth,1,scope=IMAGE_POOLING_SCOPE)
                # 插值成resize的feature map
                image_feature=tf.image.resize_bilinear(image_feature,[resize_height,resize_width],
                                                      align_corners=True)
                '''
                if isinstance(resize_height,tf.Tensor):
                    resize_height=None
                if isinstance(resize_width,tf.Tensor):
                    resize_width=None
                '''
                image_feature.set_shape([None,resize_height,resize_width,depth])
                branch_logits.append(image_feature)
            
            # step 2 对features做1x1卷积,注意此处并不是对经过pooling的image_feature做1x1卷积.
            # 融合A部分(ASPP) 需要1x1
            temp=slim.conv2d(features,depth,1,scope=ASPP_SCOPE+str(0))
            branch_logits.append(temp)
            
            # ASPP,的金字塔每层采用不同的atrous rates,此处构建这组atrous pyramid
            # 融合A部分(ASPP) 需要3x3 带artous.
            if model_option.atrous_rates:
                # 3x3卷积
                for i,rate in enumerate(model_options.atrous_rates,1):
                    scope=ASPP_SCOPE+str(i)
                    # 如果采用可分离卷积
                    if model_options.aspp_with_separable_conv:
                        aspp_features=split_separable_conv2d(
                            features,
                            filters=depth,
                            rate=rate,
                            weight_decay=weight_decay,
                            scope=scope)
                    else:
                        aspp_features=slim.conv2d(features,depth,3,rate=rate,scope=scope)
                    
                    branch_logits.append(aspp_features)
             
            # 把这些组件组合起来
            concat_logits=tf.concat(branch_logits,3) # 在通道上增加了.增加了通道
            concat_logits=slim.conv2d(
                concat_logits,depth,1,scope=CONCAT_PROJECTION_SCOPE)
            concat_logits=slim.dropout(concat_logits,keep_prob=0.9,is_training=is_training,
                                      scope=CONCAT_PROJECTION_SCOPE+'_dropout')
            
    return concat_logits,end_points
                
                    
#-------------------------------------------------------------------------------

def split_separable_conv2d(inputs,
                           filters,
                           kernel_size=3,
                           rate=1,
                           weight_decay=0.00004,
                           depthwise_weights_initializer_stddev=0.33,
                           pointwise_weights_initializer_stddev=0.06,
                           scope=None):
    """把一个separable covn2d转化成 depthwise 和 pointwise的conv2d
        depthwise_filter。一个张量，数据维度是四维[filter_height,filter_width,in_channels,channel_multiplier]，如1中所述，但是卷积深度是1
        pointwise_filter 一个张量, 维度是[1, 1, in_ch*ch_muli, out_ch]
       注意:
            该函数和tf.layers.separable_conv2d是有区别的. 该函数会在depthwise和pointwise间加上一个激活函数
        
        filters:
            是输出的个数,可理解为num_outputs
    """
    
    """注意slim.separable_conv2d的outputs参数
            如果outputs参数是None,slim.separable_conv2d会跳过point_wise阶段
            (num_outputs: The number of pointwise convolution output filters. If is
             None, then we skip the pointwise convolution stage.)
    """
    outputs=slim.separable_conv2d(
        inputs,
        None, # 只做depth_wise阶段,跳过point_wise阶段
        kernel_size=kernel_size,
        depth_multiplier=1, # DM是1
        rate=rate,
        weights_initializer=tf.truncated_normal_initializer(
            stddev=depthwise_weights_initializer_stddev
        ),
        weight_regularizer=None,
        scope=scope+'_depthwise')
    return slim.conv2d(
        outputs,# 上一层的输出,接着做1x1xfilters的point wise阶段
        filters, # 输出的filter的个数
        1,
        weight_initializer=tf.truncated_normal_initializer(
            stddev=pointwise_weights_initializer_stddev
        ),
        weight_regularizer=slim.l2_regularizer(weight_decay),
        scope=scope+'_pointwise')
    

def refine_by_decoder(features,
                      end_points,
                      decoder_height,
                      decoder_width,
                      decoder_use_separable_conv=False,
                      model_variant=None,
                      weight_decay=0.0001,
                      reuse=None,
                      is_training=False,
                      fine_tune_batch_norm=False):
    """ 添加decoder部分
    
    """
    batch_norm_params={
        'is_training':is_training and fine_tune_batch_norm,
        'decay':0.9997,
        'epsilon':1e-5,
        'scale':True,
    }
    
    with slim.arg_scope(
        [slim.conv2d,slim.separable_conv2d],
        weight_regularizer=slim.l2_regualarizer(weight_decay),
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm,
        padding='SAME',
        stride=1,
        reuse=reuse):
        with slim.arg_scope([slim.batch_norm],**batch_norm_params):
            with tf.variable_scope(DECODER_SCOPE,DECODER_SCOPE,[features]):
                feature_list=feature_extractor.networks_to_feature_maps[
                    model_variant][feature_extractor.DECODER_END_POINTS]
                if feature_list is None:
                    tf.logging.info('Not found')
                    return features
                else:
                    decoder_features=features
                    for i,name in enumerate(feature_list):
                        decoder_features_list=[decoder_features]
                        if 'mobilenet' in model_variant:
                            feature_name=name
                        else:
                            feature_name='{}/{}'.format(
                                feature_extractor.name_scope[model_variant],
                                name)
                        decoder_features_list.append(
                            slim.conv2d(
                                end_points[feature_name],
                                48,
                                1,
                                scope='feature_projection'+str(i)))
                        
                        # resize
                        for j,feature in enumerate(decoder_features_list):
                            decoder_features_list[j]=tf.image.resize_bilinear(
                                feature,[decoder_height,decoder_widht],
                                align_corners=True)
                            h=(None if isinstance(decoder_height,tf.Tensor)
                               else decoder_height)
                            w=(None if isinstance(decoder_width,tf.Tensor)
                               else decoder_width)
                            decoder_features_list[j].set_shape([None,h,w,None])
                        decoder_depth=256
                        
                        if decoder_use_separable_conv:
                            decoder_features=split_separable_conv2d(
                                tf.concat(decoder_features_list,3),
                                filters=decoder_depth,
                                rate=1,
                                weight_decay=weight_decay,
                                scope='decoder_conv0')
                            decoder_features=split_separable_conv2d(
                                tf.concat(decoder_features_list,3),
                                filters=decoder_depth,
                                rate=1,
                                weight_decay=weight_decay,
                                scope='decoder_conv1')
                        else:
                            num_convs=2
                            decoder_features=slim.repeat(
                                tf.concat(decoder_features_list,3),
                                num_convs,
                                slim.conv2d,
                                decoder_depth,
                                3,
                                scope='decoder_conv'+str(i))
                            
                return decoder_features
                            
                        
                            
    
                
        
                    
                
            
        
    
#-------------------------------------------------------------------------------
def get_branch_logits(features,
                      num_class,
                      atrous_rates=None,
                      aspp_with_batch_norm=False,
                      kernel_size=1,
                      weight_decay=0.0001,
                      reuse=None,
                      scope_suffix=''):
    """ 从模型中获得logits
        xception 后接aspp的输出是logits.
    """
    # 当aspp应用bn时,在extract_features之前就用上aspp,这里采用1x1的conv
    if aspp_with_batch_norm or atrous_rates is None:
        if atrous_rates!=1:
            #如果有bn的atrous也有,证明是aspp.需要加一个1x1的conv
            raise ValueError('kernel size must be 1')
        atrous_rates=[1]
    
    with slim.arg_scope(# slim.arg_scope作用就是我们可以预先写一些个参数,以后再调用op的时候可以不用写了,减少书写.
        [slim.conv2d],
        weight_regularizer=slim.l2_regularizer(weight_decay),
        weight_initializer=tf.truncated_normal_initializer(stddev=0.01),
        reuse=reuse):
        with tf.variable_scope(LOGITS_SCOPE_NAME,LOGITS_SCOPE_NAME,[features]):
            branch_logits=[] # 对每个atrous算一个分支,对于每个分支都存一个logits.
            for i,rate in enumerate(atrous_rates):
                scope=scope_suffix
                if i:
                    scope+='_%d'%i
                
                branch_logits.append(
                    slim.conv2d(
                        features,
                        num_class,
                        kernel_size=kernel_size,
                        rate=rate,
                        activation_fn=None,
                        normalizer_fn=None,
                        scope=scope))
    
    return tf.add_n(branch_logits)
        
    
        
        
    
    

def _get_logits(images,model_options,weight_decay=0.0001,reuse=None,is_training=False,
               find_tune_batch_norm=False):
    """生成logits网络.该网络应用到aspp,atrous spatial pyramid pooling.
    """
    # 提取features和end_points.
    features,end_points=extract_features(
        images,
        model_options,
        weight_decay=weight_decay,
        reuse=reuse,
        is_training=is_training,
        find_tune_batch_norm=find_tune_batch_norm)
    
    # 如果decoder 有特殊定义的stride.需要对decoder size做scale
    if model_option.decoder_output_stride is not None:
        if model_option.crop_size is None:
            height=tf.shape(images)[1]
            width=tf.shape(images)[2]
        else:
            # crop存在
            height,width=model_option.crop_size
        
        # 求decoder使用的size.这个是经过decoder_output_stride之后的.
        decoder_height=cal_scaled_dim_val(height,1.0/model_options.decoder_output_stride)
        decoder_width=cal_scaled_dim_val(wid,1.0/model_options.decoder_output_stride)
        
        # 对features做重新refine
        # 添加decode部分,之前的deeplab采用了crf,在deeplabV3中没有使用crf.利用sep conv2d替代.
        # 只不过这里会使用之前xception的中间产物做融合.
        features=refine_by_decoder(
            features,
            end_points,
            decoder_height=decoder_height,
            decoder_widht=decoder_width,
            decoder_use_separable_conv=model_options.decoder_use_separable_conv, # 使用离散卷积
            model_variant=model_options.model_variant,
            weight_decay=weight_decay,
            reuse=reuse,
            is_training=is_training,
            fine_tune_batch_norm=find_tune_batch_norm)

    # 获得batch的logits
    # batch_normalization作用:
    # 在激活函数之前的bn模块,它接受wx+b计算的feature作为输入.可以做到如下几点:
    # 1. 提高梯度传播数度,将所有输出归一化到0~1.避免梯度消失.
    # 2. 提高模型的收敛速度.(归一化到0~1,所有的feature都是)
    # 3. 减少模型对参数初始化的影响.(归一化到0~1)
    outputs_to_logits={}
    for output in sorted(model_options.outputs_to_num_classes):
        outputs_to_logits[output]=get_branch_logits(
            features,
            model_options.outputs_to_num_classes[output],
            model_options.atrous_rates,
            aspp_with_batch_norm=model_options.aspp_with_batch_norm, # batch normalization
            weight_decay=weight_decay,
            reuse=reuse,
            scope_suffix=output) # scope_suffix后缀
        
    return outputs_to_logits
        

    
def multi_scale_logits(images,
                       model_options,
                       image_pyramid,
                       weight_decay=0.0001,
                       is_training=False,
                       fine_tune_batch_norm=False):
    """构建logits方法
    args:
        model_options: 网络配置的定义信息.
        image_pyramid: 图像金字塔
        weight_decay: 权重衰减
    """
    if not image_pyramid:
        image_pyramid=[1.0] # list
    # crop size
    crop_height=(
        model_options.crop_size[0]
        if model_options.crop_size else tf.shape(images)[1])
    crop_width=(
        model_options.crop_size[1]
        if model_options.crop_size else tf.shape(images)[0])
    # decoder_output_stride 是在decoder单元,提炼分割结果时候使用的 input/output的比
    logits_output_stride=(
        model_options.decoder_output_stride or model_options.output_stride)
    

    logit_height=cal_scaled_dim_val(crop_height,max(1.0,max(image_pyramid))/logits_output_stride) # 这个image_pyramid其实不是很清楚它的含义,是同尺寸的images组还是downsize后images
    logit_width=cal_scaled_dim_val(crop_width,max(1.0,max(image_pyramid))/logits_output_stride)   # 2018-09-17: 这个image_pyramide是包含了一组图像缩放的fractor.并不是图像本身.
    
    outputs_to_scales_to_logits={
        k:{}
        for k in model_options.outputs_to_num_classes
    }
    
    # step 1 对于每一个缩略图
    for image_scale in image_pyramid:
        if image_scale!=1.0:
            # 不是原图,需要缩放
            # 有了缩放因子,需要计算对应的缩放尺寸
            scaled_height=cal_scaled_dim_val(crop_height,image_scale)
            scaled_width=cal_scaled_dim_val(crop_width,image_scale)
            scaled_crop_size=[scaled_height,scaled_width]
            # 有了缩放尺寸,需要对原图做缩放了
            scaled_images=tf.image.resize_bilinear(images,scaled_crop_size,align_corners=True)
            
            if model_options.crops_size:
                scaled_images.set_shape([None,scaled_height,scale_width,3]) # 如果需要crop size的话,我们把scaled_images reshape成3个chn的.
        else:
            # 原图
            scaled_crop_size=model_options.crop_size
            scaled_images=images
        
        # 用做过scale的尺寸替换参数中的crop_size,然后生成网络
        updated_options=model_options._replace(crop_size=scaled_crop_size)
        outputs_to_logits=_get_logits(
            scaled_images,
            updated_options,
            weight_decay=weight_decay,
            reuse=tf.AUTO_REUSE,
            is_training=is_training,
            fine_tune_batch_norm=fine_tune_batch_norm)
        # 此时拿到结果.对结果做一个reshape,以便和其他的scale pyramid做融合使尺寸是合理的.
        for output in sorted(outputs_to_logits):
            outputs_to_logits[output]=tf.image.resize_bilinear(
                outputs_to_logits[output],
                [logit_height,logit_width],
                align_corners=True)
            
        # 只有一层pyramid,就可以返回
        if len(image_pyramid)==1:
            for output in sorted(model_options.outputs_to_num_classes):
                # 第k个scaler fractor对应的LOGITS_SCOPE_NAME,AKA,"logits"
                outputs_to_scales_to_logits[output][LOGITS_SCOPE_NAME]=outputs_to_logits[output]
            
            return outputs_to_scales_to_logits
        
        # 如果有多个pyramid fractor,需要按照对应的标签保存 
        for output in sorted(model_options.outputs_to_num_classes):
            outputs_to_scales_to_logits[output]['logits_%.2f'%image_scale]=outputs_to_logits[output]
            
    # 把多个pyramid fractor融合
    # 需要新创建一个维度,该维度为了融合使用
    for output in model_options.outputs_to_num_classes:
        all_logits=[
            tf.expand_dims(logits,axis=4)
            for logits in outputs_to_scales_to_logits[output].values()
        ]
        # 在这个新维度上做concat( 理解为连接)
        all_logits=tf.concat(all_logits,axis=4)
        # 根据不同的融合方法采用不同的tf的融合方法
        merge_fn=(
            tf.reduce_max
            if model_options.merge_method=='max' else tf.reduce_mean)
        # 在新增维度上融合.
        outputs_to_scales_to_logits[output][MERGED_LOGITS_SCOPE]=merge_fn(all_logits,axis=4)
    
    return outputs_to_scales_to_logits


#------------------------------------------------------------------------------------------
# train


num_clones=1# 'Number of clones to deploy.')


clone_on_cpu=False# 'Use CPUs to deploy clones.')


num_replicas=1# 'Number of worker replicas.')


startup_delay_steps=15#                     'Number of training steps between replicas startup.')


num_ps_tasks=0#                     'The number of parameter servers. If the value is 0# then '                     'the parameters are handled locally by the worker.')


master=''   #'='BNS name of the tensorflow server')


task=0# 'The task ID.')


# Settings for logging.

train_logdir=None#                    'Where the checkpoint and logs are stored.')


log_steps=10 # Display logging information at every log_steps.')


save_interval_secs=1200# 'How often# in seconds# we save the model to disk.')


save_summaries_secs=600 # 'How often# in seconds# we compute the summaries.')

save_summaries_images=False#                     'Save sample inputs# labels# and semantic predictions as '                     'images to summary.')

# Settings for training strategy.

learning_policy='poly'#                  'Learning rate policy for training.')

# Use 0.007 when training on PASCAL augmented training set# train_aug. When
# fine-tuning on PASCAL trainval set# use learning rate=0.0001.
base_learning_rate=.0001#                   'The base learning rate for model training.')

learning_rate_decay_factor=0.1#                   'The rate to decay the base learning rate.')


learning_rate_decay_step=2000#                     'Decay the base learning rate at a fixed step.')

learning_power=0.9#                   'The power value used in the poly learning policy.')

training_number_of_steps=30000#                     'The number of steps used for training')

momentum=0.9# 'The momentum value to use')

# When fine_tune_batch_norm=True# use at least batch size larger than 12
# (batch size more than 16 is better). Otherwise# one could use smaller batch
# size and set fine_tune_batch_norm=False.
train_batch_size=8#                     'The number of images in each batch during training.')

# For weight_decay# use 0.00004 for MobileNet-V2 or Xcpetion model variants.
# Use 0.0001 for ResNet model variants.
weight_decay=0.00004#                   'The value of the weight decay for training.')


train_crop_size=[513,513]           # 'Image crop size [height# width] during training.')
                 

last_layer_gradient_multiplier=1.0#                   'The gradient multiplier for last layers# which is used to '                   'boost the gradient of last layers if the value > 1.')

upsample_logits=True # 'Upsample logits during training.')
# Settings for fine-tuning the network.

tf_initial_checkpoint=None#                    'The initial checkpoint in tensorflow format.')

# Set to False if one does not want to re-use the trained classifier weights.
initialize_last_layer=True#                     'Initialize the last layer.')

last_layers_contain_logits_only=False#                     'Only consider logits as last layers or not.')


slow_start_step=0#                     'Training model with small learning rate for few steps.')

slow_start_learning_rate=1e-4#                   'Learning rate employed during slow start.')

# Set to True if one wants to fine-tune the batch norm parameters in DeepLabv3.
# Set to False and use small batch size to save GPU memory.
fine_tune_batch_norm=True#                     'Fine tune the batch norm parameters or not.')

min_scale_factor=0.5#                   'Mininum scale factor for data augmentation.')

max_scale_factor=2.#                   'Maximum scale factor for data augmentation.')

scale_factor_step_size=0.25#                   'Scale factor step size for data augmentation.')

# For `xception_65`# use atrous_rates = [12# 24# 36] if output_stride = 8# or
# rates = [6# 12# 18] if output_stride = 16. For `mobilenet_v2`# use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
atrous_rates=None#                           'Atrous rates for atrous spatial pyramid pooling.')

output_stride=16#                     'The ratio of input to output spatial resolution.')
                 

# Dataset settings.
dataset_name='pascal_voc_seg'#                    'Name of the segmentation dataset.')
                 

train_split='train'#                    'Which split of the dataset to be used for training')
                 

dataset_dir=None# 'Where the dataset reside.
                 
#--train utils
_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'labels_class': ('A semantic segmentation label whose size matches image.'
                     'Its values range from 0 (background) to num_classes.'),
}
datasetDescriptor=collections.namedtuple(
    'DatasetDescriptor',
    [
        'splits_to_size',
        'name_classes', # 分类,包含背景类.例如pascal是20分类+1个背景
        'ignore_label'
    ]
)

_PASCAL_VOC=datasetDescriptor(
    splits_to_size={
        'train':2975,
        'val':500,
    },
    num_classes=19,
    ignore_label=255,
)

tfexample_decoder = slim.tfexample_decoder
dataset = slim.dataset
dataset_data_provider = slim.dataset_data_provider                 
def get_dataset(dataset_name,split_name,dataset_dir):
    """获得slim dataset实例
    """
    splite_size=_PASCAL_VOC.splits_to_size
    name_classes=_PASCAL_VOC.name_classes
    ignore_label=_PASCAL_VOC.ignore_label
    
    # file pattern
    file_pattern=os.path.join(dataset_dir,'%s-*'%split_name)
    
    # TF 解码协议
    keys_to_features={
        'image/encoded':tf.FixedLenFeature(
            (),tf.string,default_value=''),
        'image/filename':tf.FixedLenFeature((),tf.string,default_value=''),
        'image/format':tf.FixedLenFeature((),tf.string,default_value='jpeg'),
        'image/height':tf.FixedLenFeature((),tf.int64,default_value=0),
        'image/width':tf.FixedLenFeature((),tf.int64,default_value=0),
        'image/height':tf.FixedLenFeature((),tf.int64,default_value=0),
        'image/segmentation/class/encoded':tf.FixedLenFeature((),tf.string,default_value=''),
        'image/segmentation/class/format':tf.FixedLenFeature((),tf.string,default_value='png'),
    }
    items_to_handlers={
        'image':tfexample_decoder.Image(
            image_key='image/encoded',
            format_key='image/format',
            channels=3),
        'image_name':tfexample_decoder.Tensor('image/filename')
        'height':tfexample_decoder.Tensor('image/height')
        'width':tfexample_decoder.Tensor('image/width')
        'labels_class':tfexample_decoder.Image(
            image_key='image/segmentation/class/encoded',
            format_key='image/segmentation/class/format',
            channels=1),
    }
    decoder=tfexample_decoder.TFExampleDecoder(
        keys_to_features,
        items_to_handlers)
    
    return dataset.Dataset(
        data_sources=file_pattern,
        reader=rf.TFRecordReader,
        decoder=decoder,
        num_samples=splite_size[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        ignore_label=ignore_label,
        num_classes=num_classes,
        name=dataset_name,
        multi_label=True)

def resize_to_range(image,
                    label=None,
                    min_size=None,
                    max_size=None,
                    factor=None,
                    align_corners=True,
                    label_layout_is_chw=False,
                    scope=None,
                    method=tf.image.ResizeMethod.BILINEAR):
    """把图像做一个调整.
       面试问题1: 如何对图像做调整,并手写调整方法.给出min max size,
       返回: 整理好的image和label.
    """
    with tf.name_scope(scope,'resize_to_range',[image]):
        new_tensor_list=[]
        
    
                 
def input_get(dataset,
        crop_size,
        batch_size,
        min_resize_value=None,
        max_resize_value=None,
        resize_factor=None,
        min_scale_factor=1.,
        max_scale_factor=1.,
        scale_factor_step_size=0,
        num_readers=1,
        num_threads=1,
        dataset_split=None,
        is_training=True,
        model_variant=None):
    """把dataset做一个分割split
           这里分成了三步:
           1. dataset_data_provider 函数会返回raw data.
           2. 对raw data做预处理
           3. 然后利用tf对预处理data做batching.
       args:
           dataset_split: 字符串,描述当前是train还是test
    """
    data_provider=dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=num_readers,
        num_epochs=None if is_training else 1,
        shuffle=is_training)
    # get_data:
    # 拿到 image,label,image_name,height,width
    image,height,width=data_provider.get('image','height','width')
    if 'image_name' in data_provider.list_items():
        image_name,=data_provider.get('image_name')
    else:
        image_name=tf.constant('')
    if dataset_split !='test':
        label,=data_provider.get([labels_class])
    else:
        label=None
    
    # 检查label格式
    if label is not None:
        if label.shape.ndims==2:
            #[height,width]类型的.
            label=tf.expand_dims(label,2)
        elif label.shape.ndims==3 and label.shape.dims[2]==1:
            #[height,widht,1]类型的,第三个通道可以存在,但必须是1.要么就不存在好了.
            pass
        else:
            raise("Label shoud be [h,w] or [h,w,1]")
    
    # 对raw数据的pre process
    process_image=tf.cast(image,tf.float32)
    if label is not None:
        label=tf.cast(label,tf.int32)
    if min_resize_value is not None or max_resize_value is not None:
        [processed_image,label]=(
            resize_to_range(image=processed_image,
                           label=label,
                            min_size=min_resize_value,
                            min_size=min_resize_value,
                            max_size=max_resize_value,
                            factor=resize_factor, # factor的倍数+1
                            align_corners=True))
        
                 
                 
    
                 
def train():
    tf.logging.set_verbosity(tf.logging.INFO)
    config=model_deploy.DeploymentConfig(
        num_clones=num_clones,
        clone_on_cpu=clone_on_cpu,
        replica_id=task,
        num_replicas=num_replicas,
        num_ps_tasks=num_ps_tasks)
    
    clone_batch_size=train_batch_size//config_num_clones
    
    dataset=get_dataset(
        dataset_name,# 分割datasets的名字,是pascal_voc_seg还是什么.
        train_split,# 字符串,'train_split'或者'train'
        dataset_dir)# dataset路径
    tf.gfile.MakeDirs(train_logdir)
    tf.logging.info('Training on %s set',train_split)
    
    with tf.Graph().as_default() as graph:
        with tf.device(config.inputs_device()):
                samples=input_generator.get(
                # 从数据集中拿到样本
                dataset,
                train_batch_size,
                min_resize_value=min_resize_value,
                max_resize_value=max_resize_value,
                resize_factor=resize_factor,
                min_scale_factor=min_scale_factor,
                max_scale_factor=max_scale_factor,
                scale_factor_step_size=scale_factor_step_size,
                dataset_split=train_split,
                is_training=True,
                model_variant=model_variant)
            # slim.prefetch_queue生成一个queue实例.
            inputs_queue=prefetch_queue.prefetch_queue(
                samples,capacity=128*config.num_clones)
            
        with tf.device(config.variables_device()):
            global_step=tf.train.get_or_create_global_step() # 为当前图获得(有必要的话去创建)一个全局步数计数的tensor,一个graph只有一个这样的tensor.
            model_args=(inputs_queue,{
                common.OUTPUT_TYPE:dataset.num_classes
            },dataset.ignore_label)
            clones=model_deploy.create_clones(config,_build_deeplab,model_args)
            
            # 收集第一个clone的updates,可能有bn变量的update.
            first_clone_scope=config.clone_scope(0)
            update_ops=tf.get_collection(tf.Graphkeys.UPDATE_OPS,first_clone_scope)
            
        # 创建opt
        with tf.device(config.optimizer_device()):
            learing_rate=train_utils.get_module_learning_rate(
                learing_policy,
                base_learing_rate,
                learing_rate_decay_step,
                learing_rate_decay_factor,
                training_number_of_steps,
                learning_power,
                slow_start_step,
                slow_start_learing_rate)
            optimizer=tf.train.MomentumOptimizer(learing_rate,momentum)
            # add summary
        startup_delay_steps=task*startup_delay_steps
        # loss和opt
        with tf.device(config.variables_devices()):
            total_loss,grads_and_vars=model_deploy.optimize_clones(clones,optimizer)
            total_loss=tf.check_numerics(total_loss,'total loss is inf or nan')
            # summary total loss
            # 拿到最后一层的vars
            if last_layers_contain_logits_only:
                last_layers=['logits']
            else:
                last_layers=[
                    'logits',
                    'image_pooling',
                    'aspp',
                    'concat_projection',
                    'decoder',
                ]
            # 如果梯度需要按照不同的layer自定义存在.
            grad_mul=train_utils.get_model_gradient_multipliers(
                last_layers,last_layer_gradient_multiplier
            )
            
            if grad_mul:
                grads_and_vars=slim.learning.multiply_gradients(grads_and_vars,grad_mul)
            
            # 创建梯度更新的操作
            grads_update=optimizer.apply_gradients(grads_and_vars,global_step=global_step)
            update_ops.append(grads_update)
            update_op=tf.group(*update_ops) # 把这些op组合在一起.
            with tf.control_dependencies[update_op]:
                train_tensor=tf.identity(total_loss,name="train_op")
            
            session_config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
            slim.learn.train(
                train_tensor,
                logdir=train_logdir,
                log_every_n_steps=log_steps,
                master=master,
                number_of_steps=training_number_of_steps,
                is_chief=(task == 0),
                session_config=session_config,
                startup_delay_steps=startup_delay_steps,
                init_fn=train_utils.get_model_init_fn(
                    train_logdir,
                    tf_initial_checkpoint,
                    initialize_last_layer,
                    last_layers,
                    ignore_missing_vars=True),
                summary_op=summary_op,
                save_summaries_secs=save_summaries_secs,
                save_interval_secs=save_interval_secs
            )

            

                 
                  
                  
            
    

                
    
            
            
    