# log, checkpoint 저장 등에 대한 것 (default 그대로 가져옴)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        # wandb 사용 위한 추가
        dict(type='WandbLoggerHook',
            init_kwargs=dict(
                # project='object-detection',
                # entity='cv4',
                # name='38-1_cascade_original'
                project='od',
                entity='seunghyukshin',
                name='4_cascade+adamw+swinS+aug'
            ))
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
