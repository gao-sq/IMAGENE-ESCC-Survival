model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='GigaPathPatchEncoder'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=7,
        in_channels=768,
        init_cfg=None,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.0),
        dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
    _scope_='mmpretrain')
dataset_type = 'ImageNet'
data_preprocessor = dict(
    num_classes=7,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
bgr_mean = [103.53, 116.28, 123.675]
bgr_std = [57.375, 57.12, 58.395]
train_pipeline = [
    dict(type='LoadImageFromFile', _scope_='mmpretrain'),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic',
        _scope_='mmpretrain'),
    dict(type='CenterCrop', crop_size=224, _scope_='mmpretrain'),
    dict(type='PackInputs', _scope_='mmpretrain')
]
test_pipeline = [
    dict(type='LoadImageFromFile', _scope_='mmpretrain'),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic',
        _scope_='mmpretrain'),
    dict(type='CenterCrop', crop_size=224, _scope_='mmpretrain'),
    dict(type='PackInputs', _scope_='mmpretrain')
]
train_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type='CustomDataset',
        data_prefix='/work/home/haplox/data/guangdongshengyi_tissue_dataset/train',
        with_label=True,
        pipeline=train_pipeline,
        _scope_='mmpretrain'),
    sampler=dict(type='DefaultSampler', shuffle=True, _scope_='mmpretrain'))
val_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type='CustomDataset',
        # data_prefix='/opt/data/private/PycharmProjects/data/CRC-VAL-HE-7K',
        data_prefix = '/work/home/haplox/data/guangdongshengyi_tissue_dataset/val',
        with_label=True,
        pipeline=test_pipeline,
        _scope_='mmpretrain'),
    sampler=dict(type='DefaultSampler', shuffle=False, _scope_='mmpretrain'))
val_evaluator = dict(type='Accuracy', topk=(1, 5), _scope_='mmpretrain')
test_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type='CustomDataset',
        data_prefix = '/work/home/haplox/data/guangdongshengyi_tissue_dataset/test',
        with_label=True,
        pipeline=test_pipeline,
        _scope_='mmpretrain'),
    sampler=dict(type='DefaultSampler', shuffle=False, _scope_='mmpretrain'))
test_evaluator = val_evaluator
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.001,
        weight_decay=0.05,
        eps=1e-08,
        betas=(0.9, 0.999),
        _scope_='mmpretrain'),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys=dict({
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        })))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        end=2,
        convert_to_iter_based=True,
        _scope_='mmpretrain'),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-05,
        by_epoch=True,
        begin=2,
        _scope_='mmpretrain')
]
train_cfg = dict(by_epoch=True, max_epochs=8, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=1024)
default_scope = 'mmpretrain'
default_hooks = dict(
    timer=dict(type='IterTimerHook', _scope_='mmpretrain'),
    logger=dict(type='LoggerHook', interval=100, _scope_='mmpretrain'),
    param_scheduler=dict(type='ParamSchedulerHook', _scope_='mmpretrain'),
    checkpoint=dict(type='CheckpointHook', interval=1, _scope_='mmpretrain'),
    sampler_seed=dict(type='DistSamplerSeedHook', _scope_='mmpretrain'),
    visualization=dict(
        type='VisualizationHook', enable=False, _scope_='mmpretrain'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend', _scope_='mmpretrain')]
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    _scope_='mmpretrain')
log_level = 'INFO'
load_from = '/work/home/haplox/data/checkpoint/mmlab/swinV2/swinv2-base-w16_in21k-pre_3rdparty_in1k-256px_20220803-8d7aa8ad.pth'
resume = False
randomness = dict(seed=None, deterministic=False)
work_dir = '/work/home/haplox/project/guangdongshengyi/work_dirs/gigapath'
