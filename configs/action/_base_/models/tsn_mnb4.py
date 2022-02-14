# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(type='MetaNet',
                  pretrained='checkpoints/up-e/c-mnb4.pth',
                  repeats=[2, 3, 6, 6, 6, 12],
                  frozen_stages=4,
                  expansion=[1, 4, 6, 3, 2, 5],
                  channels=[32, 64, 128, 192, 192, 384],
                  final_drop=0.0,
                  mtb_type=4),
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
],
                      allow_failed_imports=False)
