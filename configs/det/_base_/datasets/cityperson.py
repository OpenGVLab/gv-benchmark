# dataset settings
dataset_type = 'CityPersonDataset'
data_root = './data/cityperson/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion',
         brightness_delta=180,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='MinIoURandomCrop',
         min_ious=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
         min_crop_size=0.1),
    dict(type='Resize',
         img_scale=[(1216, 608), (2048, 1024)],
         multiscale_mode='range',
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['img', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=(2048, 1024),
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img']),
         ])
]

classes = ('pedestrian', )
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(classes=classes,
               type=dataset_type,
               ann_file=data_root + 'train_10p_x10.json',
               img_prefix=data_root,
               pipeline=train_pipeline),
    val=dict(classes=classes,
             type=dataset_type,
             ann_file=data_root + 'val_gt_for_mmdetction.json',
             img_prefix=data_root +
             'leftImg8bit_trainvaltest/leftImg8bit/val_all_in_folder/',
             pipeline=test_pipeline),
    test=dict(classes=classes,
              type=dataset_type,
              ann_file=data_root + 'val_gt_for_mmdetction.json',
              img_prefix=data_root +
              'leftImg8bit_trainvaltest/leftImg8bit/val_all_in_folder/',
              pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mr')
