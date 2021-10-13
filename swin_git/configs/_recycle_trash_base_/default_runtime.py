checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook') # 텐서보드 사용
        dict(type='WandbLoggerHook',
            init_kwargs=dict(
                project='object-detection',
                entity='cv4',
                name='test'),
            ) # 실험이름
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
