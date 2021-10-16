## Overview

|Model|Backbone|Neck|Head|Leaderborad mAP|
|---|:---:|:---:|:---:|:---:|
|Swin-T|Swin-T|PA-FPN|Cascade-RCNN|0.599|
|Swin-S|Swin-T|FPN|Cascade-RCNN|0.631|
|Swin-B|Swin-T|FPN|Cascade-RCNN|0.629|

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
