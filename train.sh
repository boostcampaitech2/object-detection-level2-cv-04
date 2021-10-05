#!/bin/bash

# 실행은 터미널에서 
# bash {이 스크립트파일 이름} 

# ----- Custom Arg -----
# 아래 건들 필요 없이 여기부분만 커스텀하면 됩니다
# 따로 수정사항 없으면 주석처리!! ( ctrl+/ or cmd+/ )

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

# ----------------------

cmd=""

function addCmd() {

	if [ -n "$2" ]; then
  	cmd+="--$@ "
	fi
}

addCmd "train_json" ${train_json:-""}
addCmd "test_json" ${test_json:-""}
addCmd "image_root" ${image_root:-""}
addCmd "output_dir" ${output_dir:-""}
addCmd "output_eval_dir" ${output_eval_dir:-""}
addCmd "train_dataset" ${train_dataset:-""}
addCmd "test_dataset" ${test_dataset:-""}
addCmd "modelzoo_config" ${modelzoo_config:-""}
addCmd "custom_model" ${custom_model:-""}
addCmd "mapper" ${mapper:-""}
addCmd "trainer" ${trainer:-""}
addCmd "sampler" ${sampler:-""}
addCmd "classes" ${classes:-""}
addCmd "num_workers" ${num_workers:-""}
addCmd "roi_num_classes" ${roi_num_classes:-""}
addCmd "seed" ${seed:-""}
addCmd "base_lr" ${base_lr:-""}
addCmd "ims_per_batch" ${ims_per_batch:-""}
addCmd "max_iter" ${max_iter:-""}
addCmd "steps" ${steps:-""}
addCmd "gamma" ${gamma:-""}
addCmd "checkpoint_period" ${checkpoint_period:-""}
addCmd "test_eval_period" ${test_eval_period:-""}
addCmd "roi_batch" ${roi_batch:-""}
addCmd "resume" ${resume:-""}


source /opt/conda/bin/activate;
# echo python3 -m train $cmd #혹시나 어떤 입력이 들어가는지 궁금하시다면
python3 -m train $cmd