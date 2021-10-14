# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=488,
    warmup_ratio=0.001,
    periods=[1, 12, 24, 36],
    restart_weights=[1,1,0.6,0.3],
    min_lr=0
)
runner = dict(type='EpochBasedRunner', max_epochs=24)