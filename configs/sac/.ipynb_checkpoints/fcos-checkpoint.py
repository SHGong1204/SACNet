_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings

model = dict(
    type='FCOS',
    pretrained='/hy-tmp/sspnet/ssp/pretrained/resnet50-0676ba61.pth',
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
        type='FCOSHead',
        num_classes=10,
        regress_ranges=((-1, 32), (8, 64), (16, 128), (32, 256), (64, 3096)),
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        norm_on_bbox=True,
        centerness_on_reg=True,
        center_sampling=True,
        conv_bias=True,
        
        strides=[4, 8, 16, 32, 64],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0,
                    ignore_iof_thr=-1),
                allowed_border=-1,
                pos_weight=-1,
                debug=False)
test_cfg = dict(
                nms_pre=1000,
                score_thr=0.05,
                nms=dict(type='soft_nms', iou_threshold=0.3, min_score=0.05),
                max_per_img=500)
data = dict(samples_per_gpu=2, workers_per_gpu=4)

