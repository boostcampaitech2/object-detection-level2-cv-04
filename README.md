## Install

```bash
bash setup.sh
```

## Usage

- Detectron2 모듈로 모델 훈련하기

	train.sh 내 인자를 조정하여 HyperParameter와 경로 등을 수정합니다.

	```bash
	#train.sh 내부 코드, Custom Arg 부분만 커스텀하면 됩니다.

	# ----- Custom Arg -----
	# 따로 수정사항 없으면 주석처리 ( ctrl+/ or cmd+/ )

	# train_json="dataset/train.json" #(default : dataset/train.json)
	# test_json="dataset/test.json" #(default : dataset/test.json)
	# image_root="dataset/" #(default : dataset/ )
	# output_dir="output" #(default : output)
	# output_eval_dir="output" #(default : output) 

	# train_dataset="coco_trash_train" #(default : coco_trash_train)
	# test_dataset="coco_trash_test" #(default : coco_trash_test)
	modelzoo_config="COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml" #(default : COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml)
	custom_model="test1" # output에 저장될 이름입니다 (default : test1)
	mapper="base_mapper" #(default : base_mapper)
	trainer="base_trainer" #(default : base_trainer)
	sampler="custom_sampler" #(default : custom_sampler)

	# classes="General trash  Paper  Paper pack  Metal  Glass  Plastic  Styrofoam  Plastic bag  Battery  Clothing"
	num_workers="4" #(default : 4)
	roi_num_classes="10" #(default : 10)

	seed="42" #(default : 42)
	base_lr="0.001" #(default : 0.001)
	ims_per_batch="4" #(default : 4)
	max_iter="15000" #(default : 15000)
	steps="8000,12000" #(default : 8000,12000)
	gamma="0.005" #(default : 0.005)
	checkpoint_period="3000" #(default : 3000)
	test_eval_period="3000" #(default : 3000)
	roi_batch="128" #(default : 128)

	# resume="False" #(default : False)
	```


	```bash
	bash train.sh
	```

	
