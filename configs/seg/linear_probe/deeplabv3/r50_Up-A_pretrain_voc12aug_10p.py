_base_ = [
    '../../_base_/models/deeplabv3_resnet50.py',
    '../../_base_/datasets/pascal_voc12_aug_10p.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_20k.py'
]
model = dict(pretrained='checkpoints/up-a/r50.pth',
             backbone=dict(type='ResNet', frozen_stages=4),
             decode_head=dict(num_classes=21),
             auxiliary_head=dict(num_classes=21))
