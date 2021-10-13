_base_ = [ # 모델과 스케쥴러 수정
    '../../_base_/models/cascade_rcnn_r50_fpn.py',
    '../datasets/recycle_trash.py', # 쓰레기 데이터 셋 고정
    '../../_base_/schedules/schedule_2x.py',
    '../default_runtime.py' # 런타임 고정 - 초기 wandb설정만 해줌
]

model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')))