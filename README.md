# ♻️ Object Detection for Recycling Trash

</br>

Boostcourse AI Competition from [https://stages.ai/](https://stages.ai/)

</br>

## 👨‍🌾 Team

- Level 2 CV Team 4 - 무럭무럭 감자밭 🥔
- 팀 구성원: 김세영, 박성진, 신승혁, 이상원, 이윤영, 이채윤, 조성욱

</br>

## 🏆 LB Score

- Public LB: 0.698 mAP (3등/19팀)
- Private LB: 

</br>

## 🎈 Main Subject

- 바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 삶에 따라, '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 겪고 있음
- 버려지는 쓰레기 중 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문에 분리수거는 사회적 환경 부담 문제를 줄일 수 있는 방법 중 하나임 
- 따라서 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결하는데 이바지하고자함

</br>

## 🔑 Project Summary

- 여러 종류의 쓰레기 사진들을 입력값으로 받아 쓰레기의 종류와 위치를 파악하는 Object Detection 시스템 구현
- 여러가지 API(mmdetectrion & detectron2)와 framework들을 이용하여 최대한 결과의 다양성을 높이려함
    - 각 framework와 API의 이해도를 높이기 위하여 폴더 별로 구성함
- EDA: 주어진 데이터셋을 이해하기 위해 ipynb 파일로 시각화하여 학습데이터의 전체 & 클래스별 구성과 이미지들의 특징들을 파악해 프로젝트 인사이트를 높이려함
- CV Strategy: 각 클래스의 비율을 고려하여 Training Dataset과 Validation Dataset을 8대2 비율로 나누고 Validation Dataset의 검증력을 위해 동일한 이미지가 Training Set과 Validation Set에 들어가지 않도록 함
- Data Augmentation: EDA 이후 Albumentation을 이용하여 여러가지 실험을 거친 결과 대표적으로 Flip, RandomRotate90, RandomResizedCrop, RandomBrightnessContrast, HueSaturationValue, GaussNoise, Blur 등이 성능에 좋은 영향을 준 것으로 나타남
- Model
    - 1-stage: YOLOv5 (Backbone: CSP-Darknet)
    - 2-stage: Cascade R-CNN (Backbone: Swin)
- Other methods: TTA, Pseudo Labeling, No valid (using whole train dataset), Pseudo labeling, Data Preprocessing, Hyper tuning (IoU, Confidence threshold), Head (Faster R-CNN, Dynamic Faster R-CNN, Cascade R-CNN), Neck (FPN, PA-FPN, NasFPN)
- Submission: 학습된 모델을 기반으로 Test Dataset에 있는 이미지의 정답 라벨을 추론해 제출 파일을 csv format으로 만듦
- Ensemble: NMS와 WBF 모두 적용한 결과 WBF의 성능이 가장 좋게 나옴

### Dataset

- 전체 이미지 개수 : 9754장
- 10 class : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- 이미지 크기 : (1024, 1024)
- 학습데이터는 4883장, 평가데이터는 4871장으로 무작위 선정
    - 평가데이터: Public 50%, Private 50%

### Metrics

- mAP50 (Mean Average Precision)
    - Object Detection에서 사용하는 대표적인 성능 측정 방법
    - Ground Truth 박스와 Prediction 박스간 IoU(Intersection Over Union, Detector의 정확도를 평가하는 지표)가 50이 넘는 예측에 대해 True라고 판단

</br>

## 💁‍♀️ Composition

(폴더 구성 설명)
```
object-detection-level2-cv-04
├──dataset
|   ├──eda
|   |   ├──
|   |   ├──
|   |   └──
|   └──something   
├──train
|    ├──images/
|    ├──train_18class/
|    ├──val_18class/
|    └──train.csv
└──something
```

각 폴더 별 자세한 사용 설명은 폴더 내 README.md 참고
