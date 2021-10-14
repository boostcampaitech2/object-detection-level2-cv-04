# 🔎 Object Detection for Recycling Trash

Boostcourse AI Competition from [https://stages.ai/](https://stages.ai/)

## Team

- Level 2 CV Team 4 - 무럭무럭 감자밭 🥔
- 팀 구성원: 김세영, 박성진, 신승혁, 이상원, 이윤영, 이채윤, 조성욱

## LB Score

- Private LB: 0.698 mAP (3등/19팀)
- Public LB: 

## Main Subject

- 바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 삶에 따라, '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 겪고 있음
- 버려지는 쓰레기 중 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문에 분리수거는 사회적 환경 부담 문제를 줄일 수 있는 방법 중 하나임 
- 따라서 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결하는데 이바지하고자함

## Project Summary

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

### Composition

(폴더 구성 설명)