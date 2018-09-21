
# Constants

# Perform semantic segmentation predictions.
OUTPUT_TYPE = 'semantic'

# Semantic segmentation item names.
LABELS_CLASS = 'labels_class'
IMAGE = 'image'
HEIGHT = 'height'
WIDTH = 'width'
IMAGE_NAME = 'image_name'
LABEL = 'label'
ORIGINAL_IMAGE = 'original_image'

# Test set name.
TEST_SET = 'test'


min_resize_value= None 

max_resize_value= None 

resize_factor= None 

logits_kernel_size= 1 

model_variant= 'mobilenet_v2' 

image_pyramid= None 
add_image_level_feature= True 

 
image_pooling_crop_size= None 

aspp_with_batch_norm= True 

aspp_with_separable_conv= True 

multi_grid= None 

depth_multiplier= 1.0 

decoder_output_stride= None 

decoder_use_separable_conv= True 

merge_method= max



class ModelOptions():
    __slots__=() # 创建__slots__对象,目的是节省内存空间,类中会为每一个属性创建在dict中创建一个实例.通过声明__slots__,我们只为当前的声明的变量创建dict.不会对对象中新加的变量创建了.
    
    def __new__(cls,
                outputs_to_num_classes,
                crop_size=None,
                atrous_rates=None,
                output_stride=8):
        return super(ModelOptions,cls).__new__(
            cls,
            outputs_to_num_classes,
            crop_size,
            atrous_rates,
            output_stride,
            merge_method,
            add_image_level_feature,
            image_pooling_crop_size,
            aspp_with_batch_norm,
            aspp_with_separable_conv,
            multi_grid,
            decoder_output_stride,
            decoder_use_separabel_conv,
            logits_kernel_size,
            model_variant,
            depth_multiplier)
    
    def __deepcopy__(self,memo):
        return ModelOptions(
            copy.deepcopy(self.outputs_to_num_classes),
            self.crop_size,
            self.atrous_rates,
            self.output_stride)
    
    
