_base_ = ['./faster_rcnn_r50_fpn.py']

model = dict(
    type='FasterRCNN',
    backbone=dict(type='MetaNet',
                  repeats=[4, 6, 12, 12, 12, 24],
                  expansion=[1, 4, 6, 3, 2, 5],
                  channels=[128, 256, 512, 768, 768, 1536],
                  frozen_stages=4,
                  head_dim=64,
                  stem_dim=96,
                  final_drop=0.0,
                  drop_path_rate=0.3,
                  mtb_type=15,
                  return_tuple=False,
                  init_cfg=dict(type='Pretrained',
                                checkpoint='checkpoints/up-e/d-mnb15.pth')),
    neck=dict(type='FPN',
              in_channels=[256, 512, 768, 1536],
              out_channels=256,
              num_outs=5),
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