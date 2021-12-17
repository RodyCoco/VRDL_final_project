_base_ = [
	'/home/trainer/JennyHo/Visual_final/mmdetection/configs/_base_/models/faster_rcnn_r50_fpn.py',
	'/home/trainer/JennyHo/Visual_final/mmdetection/configs/_base_/datasets/coco_detection.py',
	'/home/trainer/JennyHo/Visual_final/mmdetection/configs/_base_/schedules/schedule_1x.py',
	'/home/trainer/JennyHo/Visual_final/mmdetection/configs/_base_/default_runtime.py'
]

# --> lr = 0.02



# dataset settings
dataset_type = 'CocoDataset'
classes = ('ALB', 'BET', 'DOL', 'LAG', 'SHARK', 'YFT', 'NoF', 'OTHER')



model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d')),

	roi_head=dict(
		bbox_head=dict(num_classes=8),
		),
    )


data = dict(
    train=dict(
        ann_file='/home/trainer/JennyHo/Visual_final/mmdetection/coco/annotations/train.json',
        classes=classes,
        img_prefix='/home/trainer/JennyHo/Visual_final/mmdetection/coco/images/train/'),
    val=dict(
        ann_file='/home/trainer/JennyHo/Visual_final/mmdetection/coco/annotations/val.json',
        classes=classes,
        img_prefix='/home/trainer/JennyHo/Visual_final/mmdetection/coco/images/train/'),
    test=dict(
        ann_file='/home/trainer/JennyHo/Visual_final/mmdetection/coco/annotations/test.json',
        classes=classes,
        img_prefix='/home/trainer/JennyHo/Visual_final/mmdetection/test/'))

# schedule settings
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=50)

load_from = '/home/trainer/JennyHo/Visual_final/mmdetection/checkpoints/faster_rcnn_x101_32x4d_fpn_1x_coco_20200203-cff10310.pth'
