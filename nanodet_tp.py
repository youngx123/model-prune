# -*- coding: utf-8 -*-
''' 
#Author       : xyoung
#Date         : 2024-06-29 09:27:34
#LastEditors  : kuai le jiu shi hahaha
#LastEditTime : 2024-07-08 22:10:21
'''

from functools import partial
import argparse
import copy

import torch_pruning as tp

import argparse
import os
import warnings

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import TQDMProgressBar

from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.trainer.task import TrainingTask
from nanodet.util import (
    NanoDetLightningLogger,
    cfg,
    convert_old_model,
    env_utils,
    load_config,
    load_model_weight,
    mkdir,
)

from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight


# def export_onnx(cfg, input_shape, logger):
#     if cfg.model.arch.backbone.name == "RepVGG":
#         deploy_model = model
#         from nanodet.model.backbone.repvgg import repvgg_det_model_convert
#         model = repvgg_det_model_convert(model, deploy_model)
#
#     dummy_input = torch.autograd.Variable(
#         torch.randn(1, 3, input_shape[0], input_shape[1])
#     )
#
#     torch.onnx.export(
#         model,
#         dummy_input,
#         "output_path.onnx",
#         verbose=True,
#         keep_initializers_as_inputs=True,
#         opset_version=11,
#         input_names=["data"],
#         output_names=["output"],
#     )
#     logger.log("finished exporting onnx ")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="E:/SOTA/nanodet_proj/config/legacy_v0.x_configs/RepVGG/nanodet-RepVGG-A0_416-D600-bsd.yml",
                        help="train config file path")
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()
    return args


def progressive_pruning(pruner, model, speed_up, example_inputs):
    model.eval()
    base_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    current_speed_up = 1
    time = 0
    while current_speed_up < speed_up:
        print("current speed up : ", current_speed_up)
        pruner.step(interactive=False)
        pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        current_speed_up = float(base_ops) / pruned_ops
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        time += 1
        print("prune time %d , Params: %.2f M  ,  MACs %.2f G" \
              % (time, nparams / 1e6, macs / 1e9))
        if pruner.current_step == pruner.iterative_steps:
            break
    return current_speed_up


''' 
#description: :
# method : 
#return {*}
'''


def get_pruner(model, example_inputs, imp,
               iterative_steps=None, ch_sparsity=None):
    unwrapped_parameters = []
    ignored_layers = []
    pruning_ratio_dict = {}
    # ignore output layers
    layer_num = len(model.head.gfl_cls)
    for i in range(layer_num):
        m = model.head.gfl_cls[i]
        if isinstance(m, torch.nn.Conv2d) and m.out_channels == model.head.output_channels:
            ignored_layers.append(m)

    aux_head = model.aux_head
    gfl_cls = aux_head.gfl_cls
    gfl_reg = aux_head.gfl_reg
    if isinstance(gfl_cls, torch.nn.Conv2d) and gfl_cls.out_channels == 7:
        print("add aux head2 ")
        ignored_layers.append(gfl_cls)

    if isinstance(gfl_reg, torch.nn.Conv2d) and gfl_reg.out_channels == 32:
        print("add aux head1 ")
        ignored_layers.append(gfl_reg)

    ## progressive pruning
    sparsity_learning = False
    if isinstance(imp, tp.importance.MagnitudeImportance):
        print(" im 1")
        sparsity_learning = False  # , prune.step()
        # imp = tp.importance.MagnitudeImportance(p=2)
        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs,
            importance=imp,
            iterative_steps=iterative_steps,
            pruning_ratio=ch_sparsity,
            # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            ignored_layers=ignored_layers,
            unwrapped_parameters=unwrapped_parameters,
            round_to=8,
        )
    elif isinstance(imp, tp.importance.BNScaleImportance):
        print(" im 2")
        # method == "group_slim":
        sparsity_learning = True
        #  pruner.regularize(model)  # for sparsity learning
        # imp = tp.importance.BNScaleImportance()
        # pruner_entry = partial(tp.pruner.BNScalePruner, reg=1e-5,
        #                        global_pruning=True)

        pruner = tp.pruner.GroupNormPruner(
            model,
            example_inputs,  # 用于分析依赖的伪输入
            importance=imp,  # 重要性评估指标
            iterative_steps=iterative_steps,  # 迭代剪枝，设为1则一次性完成剪枝
            ch_sparsity=0.5,  # 目标稀疏性，这里我们移除50%的通道 ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            ignored_layers=ignored_layers,  # 忽略掉最后的分类层
        )

        # pruner = pruner_entry(
        #     model,
        #     example_inputs,
        #     importance=imp,
        #     ignored_layers=ignored_layers,
        #     unwrapped_parameters=unwrapped_parameters,
        #     global_pruning=True,
        #     ch_sparsity=0.5,
        #     round_to=8,
        #     # iterative_steps=iterative_steps,
        #     # pruning_ratio=ch_sparsity,
        #     # pruning_ratio_dict=pruning_ratio_dict,
        #     # ignored_layers=ignored_layers,
        #     # unwrapped_parameters=unwrapped_parameters,
        #     # round_to=8,
        # )
    elif isinstance(imp, tp.importance.GroupNormImportance):
        print(" im 3")
        # method == "group_norm":
        sparsity_learning = True
        #  pruner.regularize(model)  # for sparsity learning
        # imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=True)
        pruner = pruner_entry(
            model,
            example_inputs,
            importance=imp,
            iterative_steps=iterative_steps,
            pruning_ratio=1,
            pruning_ratio_dict=pruning_ratio_dict,
            max_pruning_ratio=ch_sparsity,
            ignored_layers=ignored_layers,
            unwrapped_parameters=unwrapped_parameters,
        )

    """    
    elif isinstance(imp, tp.importance.GroupNormImportance):
        print(" im 4")
        # method == "group_sl":
        # sparsity_learning = True
        #  pruner.regularize(model)  # for sparsity learning
        # imp = tp.importance.GroupNormImportance(p=2, normalizer='max')  # normalized by the maximum score for CIFAR
        pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=True)
        pruner = pruner_entry(
            model,
            example_inputs,
            importance=imp,
            iterative_steps=iterative_steps,
            pruning_ratio=1,
            pruning_ratio_dict=pruning_ratio_dict,
            max_pruning_ratio=ch_sparsity,
            ignored_layers=ignored_layers,
            unwrapped_parameters=unwrapped_parameters,
        )
    elif isinstance(imp, tp.importance.GroupNormImportance):
        print(" im 4")
        # method == "growing_reg":
        # sparsity_learning = True
        # pruner.regularize(model)  # for sparsity learning
        # imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GrowingRegPruner, global_pruning=True)
        pruner = pruner_entry(
            model,
            example_inputs,
            importance=imp,
            iterative_steps=iterative_steps,
            pruning_ratio=1,
            pruning_ratio_dict=pruning_ratio_dict,
            max_pruning_ratio=ch_sparsity,
            ignored_layers=ignored_layers,
            unwrapped_parameters=unwrapped_parameters,
        )
    """
    return pruner, sparsity_learning


def get_dataset(train_dataset, val_dataset,
                batch_size, workers_per_gpu):
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        drop_last=True, shuffle=True, pin_memory=True,
        collate_fn=naive_collate, num_workers=workers_per_gpu,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, drop_last=False, pin_memory=True,
        num_workers=workers_per_gpu, collate_fn=naive_collate
    )
    return train_dataloader, val_dataloader


def prune_oneStep(trainer, task, train_dataloader, val_dataloader,
                  model_resume_path, example_inputs, logger):
    current_speed_up = progressive_pruning(task.pruner, task.model, 2, example_inputs)
    macs, nparams = tp.utils.count_ops_and_params(task.model, example_inputs)
    logger.info("after speed  Params: %.2f M  ,  MACs %.2f G" % (nparams / 1e6, macs / 1e9))
    trainer.fit(task, train_dataloader, val_dataloader, ckpt_path=model_resume_path)


def prune_iterativeSteps(trainer, task, train_dataloader, val_dataloader,
                         model_resume_path, example_inputs, logger):
    base_ops, _ = tp.utils.count_ops_and_params(task.model, example_inputs=example_inputs)

    for i in range(task.pruner.iterative_steps):
        trainer.fit_loop.min_epochs = trainer.fit_loop.max_epochs
        trainer.fit_loop.max_epochs = 10 + trainer.fit_loop.max_epochs
        task.model.eval()
        task.pruner.step(interactive=False)
        pruned_ops, _ = tp.utils.count_ops_and_params(task.model, example_inputs=example_inputs)
        current_speed_up = float(base_ops) / pruned_ops
        macs, nparams = tp.utils.count_ops_and_params(task.model, example_inputs)
        logger.info("prune time : %d , Params: %.2f M  ,  MACs %.2f G" % (i,
                                                                          nparams / 1e6, macs / 1e9))

        task.model.train()
        trainer.fit(task, train_dataloader, val_dataloader, ckpt_path=model_resume_path)


def Sparsity_Learning(task, trainer, train_dataloader,
                      val_dataloader, example_inputs, logger, model_resume_path):
    trainer.fit(task, train_dataloader, val_dataloader, ckpt_path=model_resume_path)

    logger.info("model progressive pruning")
    pruner = task.pruner

    # pruner.step()

    current_speed_up = progressive_pruning(pruner, task.model, 1.5, example_inputs)
    macs, nparams = tp.utils.count_ops_and_params(task.model, example_inputs)
    logger.info("after speed  Params: %.2f M  ,  MACs %.2f G" % (nparams / 1e6, macs / 1e9))

    logger.info("training after pruning")
    task.save_flag = -10
    task.pruner = None
    task.sparsity_learning = False
    task.avg_model = copy.deepcopy(task.model)
    task.weight_averager.state = dict()

    task.cfg.defrost()
    task.cfg.schedule.total_epochs += 40
    task.cfg.freeze()

    # train_dataloader, val_dataloader = get_dataset(train_dataset, val_dataset,
    #                                             2 * cfg.device.batchsize_per_gpu,
    #                                             cfg.device.workers_per_gpu
    #                                         )

    trainer.fit_loop.min_epochs = trainer.fit_loop.max_epochs
    trainer.fit_loop.max_epochs = 40 + trainer.fit_loop.max_epochs
    trainer.fit(task, train_dataloader, val_dataloader, ckpt_path=model_resume_path)


def main(args):
    load_config(cfg, args.config)
    if cfg.model.arch.head.num_classes != len(cfg.class_names):
        raise ValueError(
            "cfg.model.arch.head.num_classes must equal len(cfg.class_names), "
            "but got {} and {}".format(
                cfg.model.arch.head.num_classes, len(cfg.class_names)
            )
        )
    local_rank = int(args.local_rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    mkdir(local_rank, cfg.save_dir)

    logger = NanoDetLightningLogger(cfg.save_dir)
    logger.dump_cfg(cfg)

    if args.seed is not None:
        logger.info("Set random seed to {}".format(args.seed))
        pl.seed_everything(args.seed)

    logger.info("Setting up data...")
    train_dataset = build_dataset(cfg.data.train, "train")
    val_dataset = build_dataset(cfg.data.val, "test")

    evaluator = build_evaluator(cfg.evaluator, val_dataset)

    train_dataloader, val_dataloader = get_dataset(train_dataset, val_dataset,
                                                   cfg.device.batchsize_per_gpu,
                                                   0)

    logger.info("Creating model...")
    task = TrainingTask(cfg, prune=None, evaluator=evaluator)

    if "load_model" in cfg.schedule:
        ckpt = torch.load(cfg.schedule.load_model, map_location='cuda:0')
        if "pytorch-lightning_version" not in ckpt:
            warnings.warn(
                "Warning! Old .pth checkpoint is deprecated. "
                "Convert the checkpoint with tools/convert_old_checkpoint.py "
            )
            ckpt = convert_old_model(ckpt)
        load_model_weight(task.model, ckpt, logger)
        logger.info("Loaded model weight from {}".format(cfg.schedule.load_model))

    model_resume_path = (
        os.path.join(cfg.save_dir, "model_last.ckpt")
        if "resume" in cfg.schedule
        else None
    )
    if cfg.device.gpu_ids == -1:
        logger.info("Using CPU training")
        accelerator, devices, strategy, precision = (
            "cpu", None,
            None, cfg.device.precision,
        )
    else:
        accelerator, devices, strategy, precision = (
            "gpu", cfg.device.gpu_ids,
            None, cfg.device.precision,
        )

    if devices and len(devices) > 1:
        strategy = "ddp"
        env_utils.set_multi_processing(distributed=True)

    logger.info("Creating pruner...")

    # imp = tp.importance.MagnitudeImportance(p=2)  # im 1 ok
    # imp = tp.importance.BNScaleImportance()  # im 2  error
    imp = tp.importance.GroupNormImportance(p=2)    # im 3 ok

    input_size = cfg.data.train.input_size
    example_inputs = torch.randn(1, 3, input_size[0], input_size[1])

    iterative_steps = 20

    pruner, sparsity_learning = get_pruner(task.model, example_inputs, imp,
                                           iterative_steps=iterative_steps,
                                           ch_sparsity=0.5)

    logger.info("Creating pruner end...")

    logger.info("calculate base mac and params ...")
    base_macs, base_nparams = tp.utils.count_ops_and_params(task.model, example_inputs)
    logger.info("Params: %.2f M  ,  MACs : %.2f G" % (base_nparams / 1e6, base_macs / 1e9))

    logger.info("Creating pl.Trainer...")
    trainer = pl.Trainer(
        default_root_dir=cfg.save_dir,
        max_epochs=cfg.schedule.total_epochs,
        check_val_every_n_epoch=cfg.schedule.val_intervals,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=cfg.log.interval,
        num_sanity_val_steps=0,
        callbacks=[TQDMProgressBar(refresh_rate=0)],  # disable tqdm bar
        logger=logger,
        benchmark=cfg.get("cudnn_benchmark", True),
        gradient_clip_val=cfg.get("grad_clip", 0.0),
        strategy=strategy,
        precision=precision,
    )

    if not sparsity_learning:
        # pruner.step()
        ##or
        current_speed_up = progressive_pruning(pruner, task.model, 1.5, example_inputs)

        macs, nparams = tp.utils.count_ops_and_params(task.model, example_inputs)
        logger.info("after speed  Params: %.2f M  ,  MACs %.2f G" % (nparams / 1e6, macs / 1e9))

        logger.info("Fine tune training ")
        task.pruner = None
        task.sparsity_learning = False
        trainer.fit(task, train_dataloader, val_dataloader, ckpt_path=model_resume_path)
    else:
        # pruner.step()
        # current_speed_up = progressive_pruning(pruner, task.model, 1.5, example_inputs)
        task.pruner = pruner
        task.sparsity_learning = sparsity_learning
        task.avg_model = copy.deepcopy(task.model)
        logger.info(" *****Sparsity Learning  ***** ")
        Sparsity_Learning(task, trainer, train_dataloader, val_dataloader,
                          example_inputs, logger, model_resume_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
