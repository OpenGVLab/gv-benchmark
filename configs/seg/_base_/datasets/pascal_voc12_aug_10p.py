_base_ = './pascal_voc12.py'
# dataset settings
data = dict(train=dict(ann_dir=['SegmentationClass', 'SegmentationClassAug'],
                       split=[
                           'ImageSets/Segmentation/train_10p.txt',
                           'ImageSets/Segmentation/aug_10p.txt'
                       ]))
