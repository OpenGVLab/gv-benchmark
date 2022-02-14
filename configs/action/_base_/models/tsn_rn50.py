# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(type='ResNet',
                  pretrained='checkpoints/up-e/c-r50.pth',
                  depth=50,
                  frozen_stages=4,
                  norm_eval=False),
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
