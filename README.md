# โป๏ธ Object Detection for Recycling Trash

</br>

Boostcourse AI Competition from [https://stages.ai/](https://stages.ai/)

</br>

## ๐จโ๐พ Team

- Level 2 CV Team 4 - ๋ฌด๋ญ๋ฌด๋ญ ๊ฐ์๋ฐญ ๐ฅ
- ํ ๊ตฌ์ฑ์: ๊น์ธ์, ๋ฐ์ฑ์ง, ์ ์นํ, ์ด์์, ์ด์ค์, ์ด์ฑ์ค, ์กฐ์ฑ์ฑ

</br>

## ๐ LB Score

- Public LB: 0.698 mAP (3๋ฑ/19ํ)
- Private LB: 0.685 mAP (3๋ฑ/19ํ)

</br>

## ๐ Main Subject

- ๋ฐ์ผํ๋ก **๋๋ ์์ฐ, ๋๋ ์๋น**์ ์๋. ์ฐ๋ฆฌ๋ ๋ง์ ๋ฌผ๊ฑด์ด ๋๋์ผ๋ก ์์ฐ๋๊ณ  ์๋น๋๋ ์๋๋ฅผ ์ถ์ ๋ฐ๋ผ **์ฐ๋ ๊ธฐ ๋๋, ๋งค๋ฆฝ์ง ๋ถ์กฑ**๊ณผ ๊ฐ์ ์ฌํ ๋ฌธ์  ๋ฐ์
- ๋ฒ๋ ค์ง๋ ์ฐ๋ ๊ธฐ ์ค ์ ๋ถ๋ฆฌ๋ฐฐ์ถ ๋ ์ฐ๋ ๊ธฐ๋ ์์์ผ๋ก์ ๊ฐ์น๋ฅผ ์ธ์ ๋ฐ์ ์ฌํ์ฉ๋์ง๋ง, ์๋ชป ๋ถ๋ฆฌ๋ฐฐ์ถ ๋๋ฉด ๊ทธ๋๋ก ํ๊ธฐ๋ฌผ๋ก ๋ถ๋ฅ๋์ด ๋งค๋ฆฝ ๋๋ ์๊ฐ๋๊ธฐ ๋๋ฌธ์ ๋ถ๋ฆฌ์๊ฑฐ๋ ์ฌํ์  ํ๊ฒฝ ๋ถ๋ด ๋ฌธ์ ๋ฅผ ์ค์ผ ์ ์๋ ๋ฐฉ๋ฒ
- Deep Learning์ ํตํด ์ฐ๋ ๊ธฐ๋ค์ ์๋์ผ๋ก ๋ถ๋ฅํ  ์ ์๋ ๋ชจ๋ธ ๊ฐ๋ฐ 

</br>

## โ Development Environment
- GPU : Nvidia Tesla V100
- OS : Linux Ubuntu 18.04
- Runtime : Python 3.8.5
- Main Dependency : Yolov5, MMdetection, Detectron2, Pytorch 1.7.1, OpenCV 4.5.1

<br>

## ๐ Project Summary

- ์ฌ๋ฌ ์ข๋ฅ์ ์ฐ๋ ๊ธฐ ์ฌ์ง๋ค์ ์๋ ฅ๊ฐ์ผ๋ก ๋ฐ์ ์ฐ๋ ๊ธฐ์ ์ข๋ฅ์ ์์น๋ฅผ ํ์ํ๋ Object Detection
- ๋ค์ํ API([mmdetection](https://github.com/open-mmlab/mmdetection) & [detectron2](https://github.com/facebookresearch/detectron2) & [yolov5](https://github.com/ultralytics/yolov5)) ํ์ฉ    
- EDA: ์ฃผ์ด์ง ๋ฐ์ดํฐ์์ ์ดํดํ๊ธฐ ์ํด ipynb ํ์ผ๋ก ์๊ฐํํ์ฌ ํ์ต๋ฐ์ดํฐ์ ์ ์ฒด & ํด๋์ค๋ณ ๊ตฌ์ฑ๊ณผ ์ด๋ฏธ์ง๋ค์ ํน์ง๋ค์ ํ์
- CV Strategy: ๊ฐ ํด๋์ค์ ๋น์จ์ ๊ณ ๋ คํ Training Dataset๊ณผ Validation Dataset์ 8๋2 ๋น์จ๋ก ๋ถ๋ฆฌ
- Data Augmentation : Albumentation ๋ผ์ด๋ธ๋ฌ๋ฆฌ๋ฅผ ์ด์ฉ
    - Flip, RandomRotate90 : ๊ฐ์ฅ ํจ๊ณผ์ ์ธ Augmentation์ด์์ผ๋ฉฐ ์ดํ TTA์์๋ ์ฌ์ฉ๋์ด ๋์ ์ฑ๋ฅํฅ์
    - RandomResizedCrop : Flip๊ณผ ๋ง์ฐฌ๊ฐ์ง๋ก ์ฌ๋ฌ๊ฐ์ง ํฌ๊ธฐ์ ์๋ฆฐ ์ด๋ฏธ์ง๋ค์ด ๋ค์ด์ฌ ์ ์์ด์ ํด๋น Augmentation ์ ์ฉ
    - RandomBrightnessContrast, HueSaturationValue : EDA ๊ฒฐ๊ณผ ์ฌ๋ฌ๊ฐ์ง ๋ฐ๊ธฐ์ ์์์ ์๋ ฅ์ด ๋ค์ด์ฌ ์ ์์ด์ ํด๋น Augmentation์ ์ ์ฉ
    - GaussNoise, Blur : ์ด์ ์ด ์ด๊ธ๋ ์ฌ์ง์ด ์์ด ํด๋น Augmentation ์ ์ฉ
- [TTA(Test Time Augmentation)](https://inspaceai.github.io/2019/12/20/Test_Time_Augmentation_Review/) ์ ์ฉ
- Ensemble: [Weighted-Boxes-Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) WBF, IoU=0.6 ์ผ๋ก ๋ชจ๋ธ ์์๋ธ 

### Dataset

- ์ ์ฒด ์ด๋ฏธ์ง ๊ฐ์ : 9754์ฅ
- 10 class : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- ์ด๋ฏธ์ง ํฌ๊ธฐ : (1024, 1024)
- ํ์ต๋ฐ์ดํฐ๋ 4883์ฅ, ํ๊ฐ๋ฐ์ดํฐ๋ 4871์ฅ์ผ๋ก ๋ฌด์์ ์ ์ 
    - ํ๊ฐ๋ฐ์ดํฐ: Public 50%, Private 50%

### Metrics

- mAP50 (Mean Average Precision)
    - Object Detection์์ ์ฌ์ฉํ๋ ๋ํ์ ์ธ ์ฑ๋ฅ ์ธก์  ๋ฐฉ๋ฒ
    - Ground Truth ๋ฐ์ค์ Prediction ๋ฐ์ค๊ฐ IoU(Intersection Over Union, Detector์ ์ ํ๋๋ฅผ ํ๊ฐํ๋ ์งํ)๊ฐ 50์ด ๋๋ ์์ธก์ ๋ํด True๋ผ๊ณ  ํ๋จ

</br>

## ๐โโ๏ธ Composition

### Used Model
|Model|Neck|Head|Backbone|model_dir
|---|:---:|:---:|:---:|---|
|Swin|PA-FPN|Cascade-RCNN|Swin|/mmdetection|
|Swin-S|FPN|Cascade-RCNN|Swin|/mmdetection|
|Swin-B|FPN|Cascade-RCNN|Swin|/Swin-Transformer-Object-Detection|
|EfficientDet|-|-|Efficientnet|/efficientdet|
|YOLOv5x6|-|-|YOLOv5|/yolov5|

### Working Directory
```
โโโdataset
|   โโโeda
|   โโโyolov5       # dataset by yolo format
|   โโโjson files   # dataset by coco format
โโโoutput
|   โโโdetectron
|   โโโmmdet
|   โโโyolov5
โโโdetectron
โโโmmdetection
โโโSwin-Transformer-Object-Detection # swin-b
โโโefficientdet
โโโyolov5
```

๊ฐ ํด๋ ๋ณ ์์ธํ ์ฌ์ฉ ์ค๋ช์ ํด๋ ๋ด README.md ์ฐธ๊ณ 
