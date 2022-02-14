# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    # pretrained='checkpoints/metanet-b4/det_fmtb4_v1.pth.tar',
    backbone=dict(type='MetaNet',
                  repeats=[4, 6, 12, 12, 12, 24],
                  expansion=[1, 4, 6, 3, 2, 5],
                  channels=[128, 256, 512, 768, 768, 1536],
                  head_dim=64,
                  stem_dim=96,
                  final_drop=0.0,
                  drop_path_rate=0.3,
                  block_ops=['FusedMBConv3x3'] * 2 + ['MBConv3x3'] * 2 +
                  ['Block'] * 2,
                  input_size=256),
    decode_head=dict(type='ASPPHead',
                     in_channels=1536,
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
                        in_channels=768,
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
    'gvbenchmark.seg.models.backbones.metanet',
],
                      allow_failed_imports=False)
