##############################
# Backbone: Swin-T           #
# Neck:     PA-FPN           #
# Model:    Cascade R-CNN    #
# Opt:      AdamW            #
# LR:       0.0001           #
# Sch:      CosineAnnealing  #
# Epoch:    24               #
# Batch:    8                #
##############################

# merge configs
_base_ = [
    '../models/cascade_rcnn_r50_fpn.py', 
    '../datasets/dataset_all_data.py', 
    '../default_runtime.py', 
    '../schedules/schedule_2x.py', 
]

# Load pretrained Swin-T model
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'

# set model backbone to Swin-T
model = dict(
    backbone=dict(
        _delete_=True,              # dump original config's model.backbone
        type='SwinTransformer',     # use SwinTransformer
        embed_dims=96,
        depths=[2, 2, 6, 2],        # Swin-T
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),   # use pretrained model
    neck=dict(
        _delete_=True,              # dump original config's model.neck
        type='PAFPN',               # use PA-FPN
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5
        ))

# modify data.test pipeline config - for TTA
data = dict(
    test=dict(
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=True,
                flip_direction=['horizontal', 'vertical', 'diagonal'],
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]
    )
)