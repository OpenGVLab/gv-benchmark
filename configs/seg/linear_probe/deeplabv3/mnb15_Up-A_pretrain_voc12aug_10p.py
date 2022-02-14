_base_ = [
    '../../_base_/models/deeplabv3_mnb15.py',
    '../../_base_/datasets/pascal_voc12_aug_10p.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_20k.py'
]
model = dict(pretrained='checkpoints/up-a/mnb15.pth',
             backbone=dict(frozen_stages=4),
             decode_head=dict(num_classes=21),
             auxiliary_head=dict(num_classes=21))
data = dict(samples_per_gpu=2, workers_per_gpu=4)

find_unused_parameters = True
