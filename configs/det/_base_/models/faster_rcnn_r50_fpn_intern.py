_base_ = ['./faster_rcnn_r50_fpn.py']

model = dict(
    type='FasterRCNN',
    backbone=dict(frozen_stages=4,
                  init_cfg=dict(type='Pretrained',
                                checkpoint='checkpoints/up-e/d-r50.pth')),
    rpn_head=dict(
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(bbox_head=dict(
        num_classes=1,
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(rpn=dict(allowed_border=0),
                   rpn_proposal=dict(nms_across_levels=False,
                                     nms_pre=2000,
                                     nms_post=2000,
                                     max_num=2000,
                                     nms_thr=0.7,
                                     min_bbox_size=0)),
    test_cfg=dict(rpn=dict(nms_across_levels=False,
                           nms_pre=1000,
                           nms_post=1000,
                           max_num=1000,
                           nms_thr=0.7,
                           min_bbox_size=0)))
