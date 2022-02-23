_base_ = [
    '../../_base_/models/faster_rcnn_central_r50_fpn_intern.py',
    '../../_base_/datasets/voc0712_10p.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]

model = dict(backbone=dict(init_cfg=dict(
    type='Pretrained',
    checkpoint='checkpoints/up-g/r50-det-bn.pth',
)),
             roi_head=dict(bbox_head=dict(num_classes=20)))

custom_imports = dict(imports=[
    'gvbenchmark.det.models.backbones.central_model',
],
                      allow_failed_imports=False)
