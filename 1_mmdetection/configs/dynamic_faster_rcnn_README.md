## Notice

- using_file : dynamic_faster_rcnn_pafpn_swin_s.py

# train, test 공통
- cfg = Config.fromfile(' ') # cfg 파일 불러오기
- cfg.work_dir = ' ' # checkpoints 저장위치
- cfg.seed=42 # seed 고정

# train 시
- cfg.gpu_ids = [0]

# test 시
- cfg.gpu_ids = [1]
- cfg.data.test.test_mode = True
- cfg.model.train_cfg = None