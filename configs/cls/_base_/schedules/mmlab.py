# optimizer 0.01_wd_0.001_0.9
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[3000, 6000, 9000])
runner = dict(type='IterBasedRunner', max_iters=10000)
