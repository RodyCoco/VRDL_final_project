_base_ = [
	'/content/mmdetection/configs/_base_/models/faster_rcnn_r50_fpn.py',
	'/content/mmdetection/configs/_base_/datasets/coco_detection.py',
	'/content/mmdetection/configs/_base_/schedules/schedule_1x.py',
	'/content/mmdetection/configs/_base_/default_runtime.py'
]

# --> lr = 0.02



# dataset settings
dataset_type = 'CocoDataset'
classes = ('ALB', 'BET', 'DOL', 'LAG', 'SHARK', 'YFT', 'NoF', 'OTHER')



model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
	roi_head=dict(
		bbox_head=dict(num_classes=8),
		),
    )


data = dict(
    train=dict(
        ann_file='/content/mmdetection/coco/annotations/train.json',
        classes=classes,
        img_prefix='/content/mmdetection/coco/images/train/'),
    val=dict(
        ann_file='/content/mmdetection/coco/annotations/val.json',
        classes=classes,
        img_prefix='/content/mmdetection/coco/images/train/'),
    test=dict(
        ann_file='/content/mmdetection/coco/annotations/test.json',
        classes=classes,
        img_prefix='/content/mmdetection/test/'))

runner = dict(type='EpochBasedRunner', max_epochs=50)

load_from = '/content/mmdetection/checkpoints/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth'
