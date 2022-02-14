# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    # pretrained='checkpoints/metanet-b4/det_fmtb4_v1.pth.tar',
    backbone=dict(
        type='Central_Model',
        backbone_name='MTB4',
        task_names=('gv_patch', 'gv_global'),
        main_task_name='gv_global',
        trans_type='crossconvhrnetlayer',
        task_name_to_backbone={
            'gv_global': {
                'repeats': [2, 3, 6, 6, 6, 12],
                'expansion': [1, 4, 6, 3, 2, 5],
                'channels': [32, 64, 128, 192, 192, 384],
                'final_drop': 0.0,
                'block_ops': ['MBConv3x3'] * 4 + ['SABlock'] * 2,
                'input_size': 256
            },
            'gv_patch': {
                'repeats': [2, 3, 6, 6, 6, 12],
                'expansion': [1, 4, 6, 3, 2, 5],
                'channels': [32, 64, 128, 192, 192, 384],
                'final_drop': 0.0,
                'block_ops': ['MBConv3x3'] * 4 + ['SABlock'] * 2,
                'input_size': 256
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
    decode_head=dict(type='ASPPHead',
                     in_channels=384,
                     in_index=3,
                     channels=512,
                     dilations=(1, 12, 24, 36),
                     dropout_ratio=0.1,
                     num_classes=19,
                     norm_cfg=norm_cfg,
                     align_corners=False,
                     loss_decode=dict(type='CrossEntropyLoss',
                                      use_sigmoid=False,
                                      loss_weight=1.0)),
    auxiliary_head=dict(type='FCNHead',
                        in_channels=192,
                        in_index=2,
                        channels=256,
                        num_convs=1,
                        concat_input=False,
                        dropout_ratio=0.1,
                        num_classes=19,
                        norm_cfg=norm_cfg,
                        align_corners=False,
                        loss_decode=dict(type='CrossEntropyLoss',
                                         use_sigmoid=False,
                                         loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

custom_imports = dict(imports=[
    'gvbenchmark.seg.models.backbones.central_model',
],
                      allow_failed_imports=False)
