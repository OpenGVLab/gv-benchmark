_base_ = ['./faster_rcnn_r50_fpn.py']

model = dict(
    type='FasterRCNN',
    backbone=dict(type='Central_Model',
                  backbone_name='MTB4',
                  task_names=('gv_patch', 'gv_global'),
                  main_task_name='gv_global',
                  trans_type='crossconvhrnetlayer',
                  frozen_stages=4,
                  task_name_to_backbone={
                      'gv_global': {
                          'repeats': [2, 3, 6, 6, 6, 12],
                          'frozen_stages': 4,
                          'expansion': [1, 4, 6, 3, 2, 5],
                          'channels': [32, 64, 128, 192, 192, 384],
                          'final_drop': 0.0,
                          'mtb_type': 4
                      },
                      'gv_patch': {
                          'repeats': [2, 3, 6, 6, 6,
                                      12],
                          'frozen_stages': 4,
                          'expansion': [1, 4, 6, 3, 2,
                                        5],
                          'channels': [32, 64, 128, 192, 192, 384],
                          'final_drop': 0.0,
                          'mtb_type': 4
                      }
                  },
                  layer2channel={
                      'layer1': 64,
                      'layer2': 128,
                      'layer3': 192
                  },
                  layer2auxlayers={
                      'layer1': [
                          'layer1',
                      ],
                      'layer2': [
                          'layer1',
                          'layer2',
                      ],
                      'layer3': ['layer1', 'layer2', 'layer3'],
                  },
                  trans_layers=['layer1', 'layer2', 'layer3'],
                  channels=[64, 128,
                            192],
                  return_tuple=False,
                  init_cfg=dict(
                      type='Pretrained',
                      checkpoint='checkpoints/up-g/mnb4-det-bn.pth',
                  )),
    neck=dict(type='FPN',
              in_channels=[64, 128, 192, 384
                           ],
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
