#!/bin/bash

# 실행은 터미널에서 
# bash {이 스크립트파일 이름} 

# ----- Custom Arg -----
# 아래 건들 필요 없이 여기부분만 커스텀하면 됩니다
# 따로 수정사항 없으면 주석처리!! ( ctrl+/ or cmd+/ )

test_json="../dataset/test.json" #default="../dataset/test.json"
image_root="../dataset/" #default = "../dataset/"
modelzoo_config="COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml" #default="COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
model_dir="../output/detectron/10-14_faster_resx101_fpn_01/train" #default="../output/detectron/10-14_faster_resx101_fpn_01/train"
model_name="model_0000009" #default="model_0000009"
output_csv_name="det_submission" #default="det_submission"

# ----------------------

cmd=""

function addCmd() {

	if [ -n "$2" ]; then
  	cmd+="--$@ "
	fi
}

addCmd "test_json" ${test_json:-""}
addCmd "image_root" ${image_root:-""}
addCmd "modelzoo_config" ${modelzoo_config:-""}
addCmd "model_dir" ${model_dir:-""}
addCmd "model_name" ${model_name:-""}
addCmd "output_csv_name" ${output_csv_name:-""}


source /opt/conda/bin/activate;
# echo python3 -m inference $cmd #혹시나 어떤 입력이 들어가는지 궁금하시다면
python3 -m inference $cmd