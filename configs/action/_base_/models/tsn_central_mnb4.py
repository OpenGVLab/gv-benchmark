# model settings

model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='Central_Model',
        pretrained='checkpoints/up-g/mnb4-cls-bn.pth',
        backbone_name='MTB4',
        task_names=('gv_patch', 'gv_global'),
        main_task_name='gv_global',
        trans_type='crossconvhrnetlayer',
        frozen_stages=4,
        task_name_to_backbone={
            'gv_global': {
                'repeats': [2, 3, 6, 6, 6, 12],
                'frozen_stages': 4,
                'expansion': [1, 4, 6, 3, 2, 5],
                'channels': [32, 64, 128, 192, 192, 384],
                'final_drop': 0.0,
                'mtb_type': 4
            },
            'gv_patch': {
                'repeats': [2, 3, 6, 6, 6, 12],
                'frozen_stages': 4,
                'expansion': [1, 4, 6, 3, 2, 5],
                'channels': [32, 64, 128, 192, 192, 384],
                'final_drop': 0.0,
                'mtb_type': 4
            }
        },
        layer2channel={
            'layer1': 64,
            'layer2': 128,
            'layer3': 192
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
                  in_channels=1280,
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
