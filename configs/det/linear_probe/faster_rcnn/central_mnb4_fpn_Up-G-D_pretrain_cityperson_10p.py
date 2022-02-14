_base_ = [
    '../../_base_/models/faster_rcnn_central_mnb4_fpn_intern.py',
    '../../_base_/datasets/cityperson.py',
    '../../_base_/schedules/schedule_cityperson.py',
    '../../_base_/default_runtime_cityperson.py'
]

custom_imports = dict(imports=[
    'gvbenchmark.det.models.backbones.central_model',
],
                      allow_failed_imports=False)
