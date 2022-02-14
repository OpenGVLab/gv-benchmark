_base_ = [
    '../../_base_/models/faster_rcnn_mnb15_fpn_intern.py',
    '../../_base_/datasets/voc0712_10p.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]

custom_imports = dict(imports=[
    'gvbenchmark.det.models.backbones.metanet',
],
                      allow_failed_imports=False)
