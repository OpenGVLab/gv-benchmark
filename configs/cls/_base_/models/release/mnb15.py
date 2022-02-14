# model settings

model = dict(type='ImageClassifier',
             backbone=dict(type='MetaNet',
                           repeats=[4, 6, 12, 12, 12, 24],
                           frozen_stages=4,
                           expansion=[1, 4, 6, 3, 2, 5],
                           channels=[128, 256, 512, 768, 768, 1536],
                           head_dim=64,
                           stem_dim=96,
                           final_drop=0.0,
                           drop_path_rate=0.3,
                           mtb_type=15,
                           return_tuple=False,
                           init_cfg=dict(
                               type='Pretrained',
                               checkpoint='checkpoints/up-e/c-mnb15.pth',
                           )),
             head=dict(
                 type='LinearClsHead',
                 num_classes=1000,
                 in_channels=1280,
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
