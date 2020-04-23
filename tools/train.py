from __future__ import division

# -*- coding: utf-8 -*-
'''
Python中默认的编码格式是 ASCII 格式，在没修改编码格式时无法正确打印汉字，所以在读取中文时会报错。
解决方法为只要在文件开头加入 # -*- coding: UTF-8 -*- 或者 # coding=utf-8 就行了
'''

import argparse
import os
from mmcv import Config

from mmdet import __version__
from mmdet.datasets import build_dataset
from mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed)
from mmdet.models import build_detector
import torch

import ipdb


def parse_args():
    #argparse是python标准库里面用来处理****命令行参数****的库
    #ArgumentParser（）方法参数须知：一般我们只选择用description
    #description=None,    - help时显示的开始文字
    parser = argparse.ArgumentParser(description='Train a detector')
    # 改动了：将config设置为可选择参数，这样就不用键入了，可以直接在这里改路径，方便
    # add_argument向该对象中添加你要关注的命令行参数和选项
    # help		可以写帮助信息
    parser.add_argument('--config', help='train config file path',default='../configs/faster_rcnn_r50_fpn_1x.py')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    # 未找到保存断点checkpoints文件的代码？
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    #action	表示值赋予键的方式，这里用到的是bool类型
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    # type   - 指定参数类型
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    # 'LOCAL_RANK'用配置Pytorch多机多卡分布式训练
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()#获得命令行参数，实际上就是获取config配置文件，将从命令行及默认的参数传输给cfg

    #刚接触mmdetection，建议不着急看代码，可以先去把config文件夹下的py配置文件先去好好了解一下，
    #因为，要改动或者微调、以及复现论文中的精度，基本上都在config文件夹下进行修改数据。
    cfg = Config.fromfile(args.config)#读取配置文件
    
    # set cudnn_benchmark 
    # 在图片输入尺度固定时开启，可以加速.一般都是关的，只有在固定尺度的网络如SSD512中才开启
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        # 创建工作目录存放训练文件，如果不键入，会自动按照py配置文件生成对应的目录
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:    
        # 断点继续训练的权值文件，这个文件在哪里？为None就没有这一步的设置
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    # init distributed env first, since logger depends on the dist info.
    # 多机多卡时用，暂时不管
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    # 在神经网络中，参数默认是进行随机初始化的。不同的初始化参数往往会导致不同的结果，
    # 当得到比较好的结果时我们通常希望这个结果是可以复现的，在pytorch中，通过设置随机数种子也可以达到这么目的。
    # 自己在按照这种方法尝试后进行两次训练所得到的loss和误差都不同，结果并没有复现。
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    # ipdb.set_trace(context=35)
    #  搭建模型
    # 
    #  1.  build_detector()在models/builder.py里，其实就是间接调用了build()。搭建模型
    #
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    # 将训练配置传入
    #  
    #  2.  将训练配置传入，其中 build_dataset()在mmdet/datasets/builder.py里实现
    #
    train_dataset = build_dataset(cfg.data.train)#配置文件中的data字典，里的字段
    #train_dataset就是一个字典了，包含了训练时的所有参数字段。
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in checkpoints as meta data
        # 要注意的是，以前发布的模型是不存这个类别等信息的，
        # 用的默认COCO或者VOC参数，所以如果用以前训练好的模型检测时会提醒warning一下，无伤大雅
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=train_dataset.CLASSES)
    #得到数据集的类，比如coco类啊，voc类的等，都是一个类
    #都要在registry里进行注册。可在datasets文件夹下的py文件看建立模型过程。
    # add an attribute for visualization convenience 添加属性以方便可视化
    model.CLASSES = train_dataset.CLASSES   # model的CLASSES属性本来没有的，但是python不用提前声明，再赋值的时候自动定义变量
    # model的CLASSES属性本来没有的，但是python不用提前声明，再赋值的时候自动定义变量，与C++不一样
    #
    #  3.  开始训练
    #
    train_detector(
        model,
        train_dataset,
        cfg,#配置文件
        distributed=distributed,#分布式训练 true or flase
        validate=args.validate,#是否在训练中validate
        logger=logger)


if __name__ == '__main__':
    main()

