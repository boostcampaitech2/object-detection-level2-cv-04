#!/bin/bash

# 실행은 터미널에서 
# bash {이 스크립트파일 이름} 

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

cmd=""

function addCmd() {
	if [ -n "$2" ]; then
  	cmd+="--$@ "
	fi
}

addCmd "weights" ${weights:-""}
addCmd "data" ${data:-""}
addCmd "hyp" ${hyp:-""}
addCmd "epochs" ${epochs:-""}
addCmd "batch_size" ${batch_size:-""}
addCmd "img_size" ${img_size:-""}
addCmd "workers" ${workers:-""}
addCmd "project" ${project:-""}
addCmd "name" ${name:-""}
addCmd "freeze" ${freeze:-""}
addCmd "save_period" ${save_period:-""}

# source /opt/conda/bin/activate;
# echo python3 -m train $cmd #혹시나 어떤 입력이 들어가는지 궁금하시다면
python3 -m train $cmd

