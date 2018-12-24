# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: object_detection/protos/train.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from object_detection.protos import optimizer_pb2 as object__detection_dot_protos_dot_optimizer__pb2
from object_detection.protos import preprocessor_pb2 as object__detection_dot_protos_dot_preprocessor__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='object_detection/protos/train.proto',
  package='object_detection.protos',
  syntax='proto2',
  serialized_pb=_b('\n#object_detection/protos/train.proto\x12\x17object_detection.protos\x1a\'object_detection/protos/optimizer.proto\x1a*object_detection/protos/preprocessor.proto\"\xe6\x04\n\x0bTrainConfig\x12\x16\n\nbatch_size\x18\x01 \x01(\r:\x02\x33\x32\x12M\n\x19\x64\x61ta_augmentation_options\x18\x02 \x03(\x0b\x32*.object_detection.protos.PreprocessingStep\x12\x1c\n\rsync_replicas\x18\x03 \x01(\x08:\x05\x66\x61lse\x12+\n\x1dkeep_checkpoint_every_n_hours\x18\x04 \x01(\r:\x04\x31\x30\x30\x30\x12\x35\n\toptimizer\x18\x05 \x01(\x0b\x32\".object_detection.protos.Optimizer\x12$\n\x19gradient_clipping_by_norm\x18\x06 \x01(\x02:\x01\x30\x12\x1e\n\x14\x66ine_tune_checkpoint\x18\x07 \x01(\t:\x00\x12(\n\x19\x66rom_detection_checkpoint\x18\x08 \x01(\x08:\x05\x66\x61lse\x12\x14\n\tnum_steps\x18\t \x01(\r:\x01\x30\x12\x1f\n\x13startup_delay_steps\x18\n \x01(\x02:\x02\x31\x35\x12\x1f\n\x14\x62ias_grad_multiplier\x18\x0b \x01(\x02:\x01\x30\x12\x18\n\x10\x66reeze_variables\x18\x0c \x03(\t\x12 \n\x15replicas_to_aggregate\x18\r \x01(\x05:\x01\x31\x12!\n\x14\x62\x61tch_queue_capacity\x18\x0e \x01(\x05:\x03\x36\x30\x30\x12\"\n\x17num_batch_queue_threads\x18\x0f \x01(\x05:\x01\x38\x12#\n\x17prefetch_queue_capacity\x18\x10 \x01(\x05:\x02\x31\x30')
  ,
  dependencies=[object__detection_dot_protos_dot_optimizer__pb2.DESCRIPTOR,object__detection_dot_protos_dot_preprocessor__pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_TRAINCONFIG = _descriptor.Descriptor(
  name='TrainConfig',
  full_name='object_detection.protos.TrainConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='batch_size', full_name='object_detection.protos.TrainConfig.batch_size', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=32,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='data_augmentation_options', full_name='object_detection.protos.TrainConfig.data_augmentation_options', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='sync_replicas', full_name='object_detection.protos.TrainConfig.sync_replicas', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='keep_checkpoint_every_n_hours', full_name='object_detection.protos.TrainConfig.keep_checkpoint_every_n_hours', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=1000,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='optimizer', full_name='object_detection.protos.TrainConfig.optimizer', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='gradient_clipping_by_norm', full_name='object_detection.protos.TrainConfig.gradient_clipping_by_norm', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='fine_tune_checkpoint', full_name='object_detection.protos.TrainConfig.fine_tune_checkpoint', index=6,
      number=7, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='from_detection_checkpoint', full_name='object_detection.protos.TrainConfig.from_detection_checkpoint', index=7,
      number=8, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='num_steps', full_name='object_detection.protos.TrainConfig.num_steps', index=8,
      number=9, type=13, cpp_type=3, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='startup_delay_steps', full_name='object_detection.protos.TrainConfig.startup_delay_steps', index=9,
      number=10, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(15),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='bias_grad_multiplier', full_name='object_detection.protos.TrainConfig.bias_grad_multiplier', index=10,
      number=11, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='freeze_variables', full_name='object_detection.protos.TrainConfig.freeze_variables', index=11,
      number=12, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='replicas_to_aggregate', full_name='object_detection.protos.TrainConfig.replicas_to_aggregate', index=12,
      number=13, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='batch_queue_capacity', full_name='object_detection.protos.TrainConfig.batch_queue_capacity', index=13,
      number=14, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=600,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='num_batch_queue_threads', full_name='object_detection.protos.TrainConfig.num_batch_queue_threads', index=14,
      number=15, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=8,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='prefetch_queue_capacity', full_name='object_detection.protos.TrainConfig.prefetch_queue_capacity', index=15,
      number=16, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=10,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=150,
  serialized_end=764,
)

_TRAINCONFIG.fields_by_name['data_augmentation_options'].message_type = object__detection_dot_protos_dot_preprocessor__pb2._PREPROCESSINGSTEP
_TRAINCONFIG.fields_by_name['optimizer'].message_type = object__detection_dot_protos_dot_optimizer__pb2._OPTIMIZER
DESCRIPTOR.message_types_by_name['TrainConfig'] = _TRAINCONFIG

TrainConfig = _reflection.GeneratedProtocolMessageType('TrainConfig', (_message.Message,), dict(
  DESCRIPTOR = _TRAINCONFIG,
  __module__ = 'object_detection.protos.train_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.TrainConfig)
  ))
_sym_db.RegisterMessage(TrainConfig)


# @@protoc_insertion_point(module_scope)
