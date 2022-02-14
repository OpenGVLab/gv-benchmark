# model settings

model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='Central_Model',
        pretrained='checkpoints/up-g/r50-cls-bn.pth',
        backbone_name='resnet',
        task_names=('gv_patch', 'gv_global'),
        main_task_name='gv_global',
        trans_type='crossconvhrnetlayer',
        frozen_stages=4,
        task_name_to_backbone={
            'gv_global': dict(
                depth=50,
                frozen_stages=4,
                norm_eval=False,
            ),
            'gv_patch': dict(
                depth=50,
                frozen_stages=4,
                norm_eval=False,
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
    ),
    cls_head=dict(type='TSNHead',
                  num_classes=400,
                  in_channels=2048,
                  spatial_type='avg',
                  consensus=dict(type='AvgConsensus', dim=1),
                  dropout_ratio=0.4,
                  init_std=0.01),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips=None))

custom_imports = dict(imports=[
    'modelzoo.metanet',
    'modelzoo.central_model',
],
                      allow_failed_imports=False)
