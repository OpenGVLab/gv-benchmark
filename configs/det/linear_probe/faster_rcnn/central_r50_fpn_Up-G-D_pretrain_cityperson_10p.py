_base_ = [
    '../../_base_/models/faster_rcnn_central_r50_fpn_intern.py',
    '../../_base_/schedules/schedule_cityperson.py',
    '../../_base_/datasets/cityperson.py',
    '../../_base_/default_runtime_cityperson.py'
]

model = dict(backbone=dict(init_cfg=dict(
    type='Pretrained',
    checkpoint='checkpoints/up-g/r50-det-bn.pth',
)))

custom_imports = dict(imports=[
    'gvbenchmark.det.models.backbones.central_model',
],
                      allow_failed_imports=False)
