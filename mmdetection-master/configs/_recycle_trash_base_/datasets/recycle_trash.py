# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/PotatoGit/object-detection-level2-cv-04/dataset/'

# EDA 결과 255로 나눈 결과임
# RGB Mean: [0.48490459 0.46038158 0.43166834]
# RGB Standard Deviation: [0.21190031 0.20929289 0.21483885]
# 255 곱한 값
# [123.65067045 117.3973029  110.0754267 ]
# [54.03457905 53.36968695 54.78390675]
# 기본
# img_norm_cfg = dict(
#     # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True) # 원본 pixel 값 mean, std
#     mean=[123.65067045, 117.3973029, 110.0754267], std=[54.03457905, 53.36968695, 54.78390675], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True), # 모델별로 맞추어 사이즈 변경 (원본 사이즈 넣어놓음)
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1024, 1024),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]

# augmentation 많은 것
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
    dict(type='RandomResizedCrop',height=512, width=512, scale=(0.5, 1.0), p=0.5),
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
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
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
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),

]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
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

# 윤영님이 주신 5개 stratified k-fold crossvalidation 중 일부를 이용하여 검증 진행 (0 ~ 4 까지 바꿔줄 수 있음)
data = dict(
    samples_per_gpu=8, # 2개는 넘 적소
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train_0.json', # 0 번째 이용
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'valid_0.json', # 0 번째 이용
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
