## Install

```bash
pip install effdet
```

## Notice

### efficientdet_size_up.ipynb file
- efficientdet 512 사이즈 예측된 거 키우기
- 이미 생성된 csv에 대해서 키워줄때 사용 (코드 내 이미 수정 반영되어있음)

### efficientdet3(label_modify)
- 슬랙에 문제 제기된 General_trash를 못잡는 오류를 해결하여 학습한 모델

## Performance(Public LB)

- efficientdet0_epoch100 : 0.294
- efficientdet0_epoch300 : 0.279
- efficientdet3(basecode) : 0.32
- efficientdet3(label_modify) : 0.357
- efficientdet4 : 0.262