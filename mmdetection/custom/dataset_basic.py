# dataset settings
dataset_type = 'CocoDataset' # trash 데이터도 coco dataset 타입임!
data_root = '../dataset/'

# 기존 coco dataset은 80개의 클래스가 기본적으로 선언되어 있음!
# 따라서 우리 데이터셋에 맞춰 클래스를 다시 선언해줘야 함!
classes = ['General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# train datset이 어떤 파이프라인을 거쳐서 augmenatation이 될지에 대한 부분
train_pipeline = [
    dict(type='LoadImageFromFile'), # 이미지를 파일로부터 불러오는 것
    dict(type='LoadAnnotations', with_bbox=True), # annotation 파일 불러오는 것 (with_bbox를 넣음으로써 box도 가져옴)
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True), # 여기서부터 augmentation에 대한 것 (trash 이미지는 1024x1024)
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32), # 이미지에 padding 처리 할 것인지에 대한 것
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

# test_pipeline은 LoadAnnotations를 안함!
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024), # trash data에 맞게 사이즈 변경
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

# 실제 config가 작성되는 곳
data = dict(
    samples_per_gpu=4, # gpu 당 batch 사이즈
    workers_per_gpu=2, # dataloader의 num_workers와 동일
    # train 데이터가 어떤 파이프라인 거칠지, 데이터셋은 어디에 있는지에 대한 부분
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'split/train_0.json', # annotation file명 지정
        img_prefix=data_root, # 이미지 파일이 들어있는 디렉토리 주소
        classes=classes,
        pipeline=train_pipeline), # 앞서 선언한 train_pipeline 넘겨줌
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'split/valid_0.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')
