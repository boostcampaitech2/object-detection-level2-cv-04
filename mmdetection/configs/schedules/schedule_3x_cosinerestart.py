# optimizer
optimizer = dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=611, # 1epoch
    warmup_ratio=0.001,
    periods = [1, 12, 12, 12],
    restart_weights = [1, 1, 0.5, 0.5],
    min_lr=0)
runner = dict(type='EpochBasedRunner', max_epochs=36)