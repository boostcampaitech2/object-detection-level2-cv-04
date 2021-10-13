# optimizer
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None) # grad_clip : 기울기 폭주를 막기 위해 임계값 넘지 않도록 값을 자르는 것 (임계치만큼 크기 감소)
# gradient clipping 관련 참고자료 (https://wikidocs.net/61375)

# learning policy
lr_config = dict(
    policy='step', # 어떤 scheduler 사용할 것인지
    warmup='linear', # scheduler 및 warmup 관련 참고자료 (https://ai4nlp.tistory.com/8) / (https://norman3.github.io/papers/docs/bag_of_tricks_for_image_classification.html)
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11]) # ?? 정확한 효과는 뭘까??

runner = dict(type='EpochBasedRunner', max_epochs=24) # 몇 epoch으로 돌지를 설정

"""
1x : runner의 max_epochs = 12
2x : runner의 max_epochs = 24
20e : runner의 max_epochs = 20 (20epochs 의미)
"""
