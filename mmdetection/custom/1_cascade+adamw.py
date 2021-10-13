# config 파일들이 각가 떨어져 있기 때문에 합쳐주는 과정 필요!
_base_ = [
    'model/cascade_rcnn_r50_fpn_trash.py',
    'dataset_basic.py',
    'default_runtime.py',
    'schedule.py'
]
