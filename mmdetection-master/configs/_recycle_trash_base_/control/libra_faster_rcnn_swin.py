_base_ = [ # 모델과 스케쥴러 수정
    '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../datasets/recycle_trash.py', # 쓰레기 데이터 셋 고정
    '../../_base_/schedules/schedule_2x.py',
    '../default_runtime.py' # 런타임 고정 - 초기 wandb설정만 해줌
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
model = dict(
    backbone=dict(
        _delete_=True, # 지울게 없음
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
    neck=[
        dict(
            type='FPN',
            in_channels=[96, 192, 384, 768],
            # in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        dict(
            type='BFP',
            in_channels=256,
            num_levels=5,
            refine_level=2,
            refine_type='non_local')
    ],
    roi_head=dict(
        bbox_head=dict(
            loss_bbox=dict(
                _delete_=True,
                type='BalancedL1Loss',
                alpha=0.5,
                gamma=1.5,
                beta=1.0,
                loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(sampler=dict(neg_pos_ub=5), allowed_border=-1),
        rcnn=dict(
            sampler=dict(
                _delete_=True,
                type='CombinedSampler',
                num=512,
                pos_fraction=0.25,
                add_gt_as_proposals=True,
                pos_sampler=dict(type='InstanceBalancedPosSampler'),
                neg_sampler=dict(
                    type='IoUBalancedNegSampler',
                    floor_thr=-1,
                    floor_fraction=0,
                    num_bins=3)))))
