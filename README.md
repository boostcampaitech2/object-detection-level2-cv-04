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
- Private LB: 0.685 mAP (3등/19팀)

</br>

## 🎈 Main Subject

- 바야흐로 **대량 생산, 대량 소비**의 시대. 우리는 많은 물건이 대량으로 생산되고 소비되는 시대를 삶에 따라 **쓰레기 대란, 매립지 부족**과 같은 여러 사회 문제를 겪고 있음
- 버려지는 쓰레기 중 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문에 분리수거는 사회적 환경 부담 문제를 줄일 수 있는 방법
- 사진에서 쓰레기를 인식하는 모델을 만들어 이러한 문제점을 해결하는데 이바지하고자함

</br>

## 🔑 Project Summary

- 여러 종류의 쓰레기 사진들을 입력값으로 받아 쓰레기의 종류와 위치를 파악하는 Object Detection
- 여러가지 API([mmdetection](https://github.com/open-mmlab/mmdetection) & [detectron2](https://github.com/facebookresearch/detectron2) & [yolov5](https://github.com/ultralytics/yolov5))내 저장된 이용하여 단일모델 출력의 다양성을 상승
    - 각 framework와 API의 이해도를 높이기 위하여 폴더 별로 구성
- EDA: 주어진 데이터셋을 이해하기 위해 ipynb 파일로 시각화하여 학습데이터의 전체 & 클래스별 구성과 이미지들의 특징들을 파악하여 프로젝트 인사이트 향상
- CV Strategy: 각 클래스의 비율을 고려한 Training Dataset과 Validation Dataset을 8대2 비율로 분리
- Data Augmentation : Albumentation 라이브러리를 이용
    - Flip, RandomRotate90 : 가장 효과적인 Augmentation이였으며 이후 TTA에서도 사용되어 높은 성능향상
    - RandomResizedCrop : Flip과 마찬가지로 여러가지 크기와 잘린 이미지들이 들어올 수 있어서 해당 Augmentation 적용
    - RandomBrightnessContrast, HueSaturationValue : EDA 결과 여러가지 밝기와 색상의 입력이 들어올 수 있어서 해당 Augmentation을 적용
    - GaussNoise, Blur : 초점이 어긋난 사진이 있어 해당 Augmentation 적용
- [TTA](https://inspaceai.github.io/2019/12/20/Test_Time_Augmentation_Review/)
- Ensemble: [Weighted-Boxes-Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) 라이브러리를 사용하여 단일 모델들로 0.601~0.629 사이의 mAP점수들을 최대 **0.698 mAP**까지 향상 

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
