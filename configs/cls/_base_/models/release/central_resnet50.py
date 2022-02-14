# model settings

model = dict(type='ImageClassifier',
             backbone=dict(type='Central_Model',
                           backbone_name='resnet',
                           task_names=('gv_patch', 'gv_global'),
                           main_task_name='gv_global',
                           trans_type='crossconvhrnetlayer',
                           frozen_stages=4,
                           task_name_to_backbone={
                               'gv_global':
                               dict(
                                   depth=50,
                                   frozen_stages=4,
                                   num_stages=4,
                                   out_indices=(3, ),
                                   style='pytorch',
                               ),
                               'gv_patch':
                               dict(
                                   depth=50,
                                   frozen_stages=4,
                                   num_stages=4,
                                   out_indices=(3, ),
                                   style='pytorch',
                               ),
                           },
                           layer2channel={
                               'layer1': 256,
                               'layer2': 512,
                               'layer3': 1024
                           },
                           layer2auxlayers={
                               'layer1': [
                                   'layer1',
                               ],
                               'layer2': [
                                   'layer1',
                                   'layer2',
                               ],
                               'layer3': ['layer1', 'layer2', 'layer3'],
                           },
                           trans_layers=['layer1', 'layer2', 'layer3'],
                           channels=[64, 128, 192],
                           return_tuple=False,
                           init_cfg=dict(
                               type='Pretrained',
                               checkpoint='checkpoints/up-g/r50-cls-bn.pth',
                           )),
             head=dict(
                 type='LinearClsHead',
                 num_classes=1000,
                 in_channels=2048,
                 init_cfg=dict(
                     type='Kaiming',
                     a=2.23606,
                     mode='fan_out',
                     nonlinearity='relu',
                     distribution='uniform',
                 ),
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, 5),
             ))
