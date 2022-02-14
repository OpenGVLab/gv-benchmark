# model settings
model = dict(type='ImageClassifier',
             backbone=dict(type='MetaNet',
                           repeats=[2, 3, 6, 6, 6, 12],
                           frozen_stages=4,
                           expansion=[1, 4, 6, 3, 2, 5],
                           channels=[32, 64, 128, 192, 192, 384],
                           final_drop=0.0,
                           mtb_type=4,
                           return_tuple=False,
                           init_cfg=dict(
                               type='Pretrained',
                               checkpoint='checkpoints/up-e/c-mnb4.pth',
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
