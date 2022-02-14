# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='MetaNet',
        pretrained='checkpoints/up-e/c-mnb15.pth',
        repeats=[4, 6, 12, 12, 12, 24],
        frozen_stages=4,
        expansion=[1, 4, 6, 3, 2, 5],
        channels=[128, 256, 512, 768, 768, 1536],
        head_dim=64,
        stem_dim=96,
        final_drop=0.0,
        drop_path_rate=0.3,
        mtb_type=15,
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
],
                      allow_failed_imports=False)
