_base_ = [
    './retinanet_r50_fpn.py',
    './dataset.py',
    './default_runtime.py'
]
cudnn_benchmark = True
# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),
    neck=dict(type='NASFPN', stack_times=7, norm_cfg=norm_cfg),
    bbox_head=dict(type='RetinaSepBNHead', num_ins=5, norm_cfg=norm_cfg),
    # training and testing settings
    train_cfg=dict(assigner=dict(neg_iou_thr=0.5)))

# optimizer
optimizer = dict(
    type='AdamW',
    betas=(0.9, 0.999),
    lr=1e-4,
    weight_decay=0.0001,
    )
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)
runner = dict(type='EpochBasedRunner', max_epochs=50)
