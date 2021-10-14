## Notice

- using_file : libra_faster_rcnn_swin.py

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


## Pseudo Labeling
- labeled json과 예측된 submission file을 Pseudo_labeling.py를 이용하여 합쳐줌(코드 내 경로 수정 필요하며, json이 생성됨)
- recycle_trash.py의 train 부분 annotation을 생성된 json에 맞춰 수정
- libra_faster_rcnn_swin.py의 init_cfg을 None으로 수정
- 학습시 checkpoint load
```bash
checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
```