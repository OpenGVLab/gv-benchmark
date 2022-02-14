_base_ = [
    '../_base_/models/release/central_resnet50.py',
    '../_base_/datasets/flower_10p.py', '../_base_/schedules/mmlab.py',
    '../_base_/default_runtime.py', '../_base_/custom_import.py'
]
model = dict(head=dict(num_classes={{_base_.dataset_num_classes}}))
