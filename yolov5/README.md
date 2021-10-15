## Overview

YOLOv5을 이용한 모델 훈련입니다.  
주로 **Yolov5l6, Yolov5x6**을 사용했습니다. 

|Model|리더보드 mAP|Valid mAP|
|---|:---:|:---:|
|YOLOv5l6|0.564|0.629|
|YOLOv5x6|0.592|0.634|
|YOLOv5x6_withTTA|0.612|0.634|

## Install

  ```
  # pip install -r requirements.txt
  ```

## Usage

- YOLOv5 모듈로 train 하기

	train.sh 내 인자를 조정하여 HyperParameter와 경로 등을 수정합니다.
  (현재는 YOLOv5(ver.6)으로 적용되어 있습니다.)

	```bash
	#train.sh 내부 코드, Custom Arg 부분만 커스텀하면 됩니다.

	# ----- Custom Arg -----
  # 아래 건들 필요 없이 여기부분만 커스텀하면 됩니다
  # 따로 수정사항 없으면 주석처리!! ( ctrl+/ or cmd+/ )

  weights="yolov5x6.pt" #(default : ROOT / 'yolov5s.pt')
  data="trash.yaml" #(default=ROOT / 'data/coco128.yaml')
  hyp="hyp.p6.yaml" #(default=ROOT / 'data/hyps/hyp.scratch.yaml')
  epochs="100" #(default=300)
  batch_size="12" #(default=16)
  img_size="1024" #(default=640)
  # workers="8" #(default=8)
  project="../output/yolov5/" # project_dir (default=ROOT / 'runs/train')
  # name="exp" # experiment_name (default='exp')
  # freeze="0" # Number of layers to freeze (default=0)
  # save_period="5" # Save checkpoint every x epochs (default=-1)
  # ----------------------
	```

	```bash
	bash train.sh
	```

- YOLOv5 모듈로 inference 하기

	inference.ipynb 내부 코드, exp_name 부분 커스텀하면 됩니다.
