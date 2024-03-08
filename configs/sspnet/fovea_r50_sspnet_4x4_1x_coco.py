_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='FOVEA_SPPNet',
    pretrained='/root/fovea/sspnet/epoch_8.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='SSFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        # start_level=1,
        num_outs=5,
        #add_extra_convs='on_input'
    ),
    bbox_head=dict(
        type='FoveaHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[4, 8, 16, 32, 64],
        base_edge_list=[8, 16, 32, 64, 128],  # [16, 32, 64, 128, 256]
        scale_ranges=((1, 32), (8, 64), (16, 128), (32, 256), (64, 512)),
        sigma=0.4,
        with_deform=False,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.50,
            alpha=0.4,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)))
# training and testing settings
train_cfg = dict()
test_cfg = dict(
    nms_pre=1000,
    score_thr=0.05,
    nms=dict(type='soft_nms', iou_threshold=0.3, min_score=0.05),
    max_per_img=500)
data = dict(samples_per_gpu=4, workers_per_gpu=4)
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
