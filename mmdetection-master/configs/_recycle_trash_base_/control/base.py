_base_ = [ # 모델과 스케쥴러 수정
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../datasets/recycle_trash.py', # 쓰레기 데이터 셋 고정
    '../../_base_/schedules/schedule_2x.py',
    '../default_runtime.py' # 런타임 고정 - 초기 wandb설정만 해줌
]

# Resnet 101로 변환
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))