_base_ = [
    '../model/cascade_rcnn_r50_fpn_trash.py',
    '../dataset/dataset_basic.py',
    '../default_runtime.py',
    '../schedule/schedule.py'
]

# 기존 cascade_rcnn에 backbone을 swin으로 변경 (swin-T)
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
model = dict(
    backbone=dict(
        _delete_=True, # 기존 사용하던 type인 resnet을 지우고, swintransformer 사용하겠다는 것 (다른 config에서도 사용 가능!)
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768])) 
    # neck 채널은 논문을 통해 알 수 있음!
    # 기존 neck의 in_channels=[256, 512, 1024, 2048]이었음! 
    # backbone마다 feature map 채널이 다름으로 유의해서 backbone custom하자!!

