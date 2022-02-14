_base_ = [
    '../../_base_/models/deeplabv3_central_resnet50.py',
    '../../_base_/datasets/pascal_voc12_aug_10p.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_20k.py'
]
model = dict(backbone=dict(frozen_stages=4,
                           task_name_to_backbone={
                               'gv_global': dict(frozen_stages=4),
                               'gv_patch': dict(frozen_stages=4)
                           }),
             init_cfg=dict(
                 type='Pretrained',
                 checkpoint='checkpoints/up-g/r50-det-bn.pth',
             ),
             decode_head=dict(num_classes=21),
             auxiliary_head=dict(num_classes=21))
data = dict(samples_per_gpu=2, workers_per_gpu=4)

find_unused_parameters = True
