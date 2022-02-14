_base_ = ['./default_runtime.py']
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
work_dir = './work_dirs/'
