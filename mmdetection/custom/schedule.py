# optimizer
# optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.05)
# optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer = dict(
    # _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

# grad_clip : 기울기 폭주를 막기 위해 임계값 넘지 않도록 값을 자르는 것 (임계치만큼 크기 감소)
# gradient clipping 관련 참고자료 (https://wikidocs.net/61375)
optimizer_config = dict(grad_clip=None)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
# scheduler 및 warmup 관련 참고자료 (https://ai4nlp.tistory.com/8) / (https://norman3.github.io/papers/docs/bag_of_tricks_for_image_classification.html)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])

# CosineAnnealing
# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 10,
#     min_lr_ratio=1e-5)

runner = dict(type='EpochBasedRunner', max_epochs=24)
