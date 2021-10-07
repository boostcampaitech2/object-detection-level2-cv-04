# optimizer
optimizer = dict(
    type='AdamW',
    betas=(0.9, 0.999),
    lr=1e-4,
    weight_decay=0.0001,
    )
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=50)
