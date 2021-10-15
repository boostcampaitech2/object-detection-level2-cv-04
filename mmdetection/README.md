## Overview

|Model|Backbone|Neck|Leaderborad mAP|
|---|:---:|:---:|:---:|
|Cascade-RCNN|Swin-T|PA-FPN|0.599|
|Cascade-RCNN|Swin-S|FPN|0.631|
|Cascade-RCNN|Swin-B|FPN|0.629|

## Install mmdetection
```bash
git clone https://github.com/open-mmlab/mmdetection
cd mmdetection
pip install openmim
mim install mmdet
```

## Train
```
cd mmdetection
python tools/train.py configs/finals/[experiment_name].py
```
all_data 실험의 경우, --no-validate 옵션을 주어 실행해야 합니다. 
