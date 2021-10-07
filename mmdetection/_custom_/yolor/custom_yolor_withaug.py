# python mmdetection/tools/train.py mmdetection/_custom_/yolor/custom_yolor_withaug.py
_base_ = '../yolor/sparse_rcnn_r50_fpn_coco.py'

num_proposals = 300
model = dict(
    rpn_head=dict(num_proposals=num_proposals),
    test_cfg=dict(
        rpn=None, rcnn=dict(max_per_img=num_proposals)))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(
    type='OneOf',
    transforms=[
        dict(type='Flip',p=1.0),
        dict(type='RandomRotate90',p=1.0)
    ],
    p=0.5),
    dict(type='RandomBrightnessContrast',brightness_limit=0.1, contrast_limit=0.15, p=0.5),
    dict(type='HueSaturationValue', hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=10, p=0.5),
    dict(type='GaussNoise', p=0.3),
    dict(
    type='OneOf',
    transforms=[
        dict(type='Blur', p=1.0),
        dict(type='GaussianBlur', p=1.0),
        dict(type='MedianBlur', blur_limit=5, p=1.0),
        dict(type='MotionBlur', p=1.0)
    ],
    p=0.1)
]

# augmentation strategy originates from DETR.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1024), (512, 1024), (544, 1024), (576, 1024),
                           (608, 1024), (640, 1024), (672, 1024), (704, 1024),
                           (736, 1024), (768, 1024), (800, 1024)],
                multiscale_mode='value',
                keep_ratio=True)
                ],
                [
                    dict(
                        type='Resize',
                        img_scale=[(400, 1024), (500, 1024), (600, 1024)],
                        multiscale_mode='value',
                        keep_ratio=True),
                    dict(
                        type='RandomCrop',
                        crop_type='absolute_range',
                        crop_size=(384, 600),
                        allow_negative_crop=True),
                    dict(
                        type='Resize',
                        img_scale=[(480, 1024), (512, 1024), (544, 1024),
                                    (576, 1024), (608, 1024), (640, 1024),
                                    (672, 1024), (704, 1024), (736, 1024),
                                    (768, 1024), (800, 1024)],
                        multiscale_mode='value',
                        override=True,
                        keep_ratio=True)
                        ]]),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True
        ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

data = dict(train=dict(pipeline=train_pipeline))

