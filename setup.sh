#!/bin/bash

# 가상환경 설정
source /opt/conda/bin/activate;

# detectron2 설치 후 설치폴더 제거
git clone https://github.com/facebookresearch/detectron2;
python -m pip install -e detectron2;
# rm -rf detectron2;

# 의존성 라이브러리 설치
pip3 install -r requirements.txt;
