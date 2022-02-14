_base_ = [
    '../_base_/models/release/mnb4.py', '../_base_/datasets/kinetics_10p.py',
    '../_base_/schedules/mmlab.py', '../_base_/longer_runtime.py',
    '../_base_/custom_import.py'
]

model = dict(head=dict(num_classes={{_base_.dataset_num_classes}}))
