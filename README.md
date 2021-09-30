## Install

1. Detectron2 설치가 선행되어야 합니다!  
	```bash
	source /opt/conda/bin/activate

	git clone https://github.com/facebookresearch/detectron2
	python -m pip install -e detectron2
	```

2. Install Dependency Library
	```bash
	pip3 install -r requirements.txt
	```

## Usage

- 모델 훈련하기

	```bash
	python3 -m train
	```