# 모듈 import
import wandb
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset, replace_ImageToTensor)

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# 작성한 config file 들고오기 (수정 1)
cfg = Config.fromfile('../../configs/_recycle_trash_base_/control/swin_b_cascade.py')

# dataset config 수정
cfg.data.train.classes = classes
cfg.data.val.classes = classes
cfg.data.test.classes = classes

# 모델에 맞게 수정해주어야할 부분들 - 아래있는건 faster_rcnn 기준 (수정 2)
# cfg.data.train.pipeline[2]['img_scale'] = (512,512) # Resize
# cfg.data.val.pipeline[1]['img_scale'] = (512,512) # Resize
# cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize

# 결과 저장 위치 (수정 3)
cfg.work_dir = './check_point/64_swin_b_cascade_trainsetAll_3x'

# 수정 4 runtime 내 실험 이름
cfg.log_config.hooks[1].init_kwargs['name'] = '64_swin_b_cascade_trainsetAll_3x'
# print(cfg.model)
# 고정
cfg.data['samples_per_gpu'] = 4
# cfg.model.roi_head.bbox_head.num_classes = 10 # 이미 클래스 바꿔 놓음
cfg.optimizer_config.grad_clip = None
cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1) # 저장 간격, 최대 개수 변경하고싶을 시
cfg.seed = 42 # 행운의 숫자 42
cfg.gpu_ids = [0] # gpu 1개 이용
cfg.data.train.ann_file = '/opt/ml/detection/PotatoGit/object-detection-level2-cv-04/dataset/train.json'

# build_dataset
datasets = [build_dataset(cfg.data.train)]

# 모델 build 및 pretrained network 불러오기
model = build_detector(cfg.model)
model.init_weights()

# 모델 학습
train_detector(model, datasets[0], cfg, distributed=False, validate=False)

import requests

def send_message_to_slack(text):
    url = "https://hooks.slack.com/services/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" # slack 알람 만들어주어야함 자신꺼에 맞게
    payload = { "text" : text }
    requests.post(url, json=payload)
    
send_message_to_slack('finish')