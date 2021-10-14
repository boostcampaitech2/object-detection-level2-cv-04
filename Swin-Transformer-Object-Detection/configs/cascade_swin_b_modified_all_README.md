## Notice

- using_file : swin_b_cascade.py

# train, test 공통
- cfg = Config.fromfile(' ') # cfg 파일 불러오기
- cfg.work_dir = ' ' # checkpoints 저장위치
- cfg.seed=42 # seed 고정

# train 시
- cfg.gpu_ids = [0]
- valid부분을 False로 학습 (바뀐 데이터셋을 모두 사용하므로)

# test 시
- cfg.gpu_ids = [1]
- cfg.data.test.test_mode = True
- cfg.model.train_cfg = None