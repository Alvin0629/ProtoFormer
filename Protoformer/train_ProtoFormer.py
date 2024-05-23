from __future__ import print_function, division
import sys
# sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch.backends.cudnn as cudnn
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from loguru import logger as loguru_logger
from torch.utils.data import DataLoader
from core import optimizer
import evaluate_Protoformer as evaluate
import core.datasets as datasets

from core.loss import sequence_loss_depth
from core.loss import sequence_loss_flow

from core.optimizer import fetch_optimizer
from core.utils.misc import process_cfg
#from loguru import logger as loguru_logger
import core.utils.init_distribute as dist_util

from core.utils.logger import Logger

from core.Protoformer import build_Protoformer

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass
        
gpu0 = torch.device('cuda:0')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(cfg):
    model = nn.DataParallel(build_Protoformer(cfg), device_ids=args.gpus)
    loguru_logger.info("Parameter Count: %d" % count_parameters(model))

    if cfg.restore_ckpt is not None:
        print("[Loading ckpt from {}]".format(cfg.restore_ckpt))

        if cfg.stage == 'depth_VKITTI_pretrain':   # depth training start from pretrained flow encoder!!!
            complete_state_dict = torch.load(cfg.restore_ckpt, map_location=gpu0)
            # Filter out keys starting from 'module.memory_decoder.update_block'
            filtered_state_dict = {k: v for k, v in complete_state_dict.items() if not k.startswith('module.flow_head')}
            # Load the filtered state dictionary into the model, with strict=False to allow for missing keys
            model.load_state_dict(filtered_state_dict, strict=False)
            
        elif cfg.stage == 'things' or cfg.stage == 'sintel' or cfg.stage == 'kitti' or cfg.stage == 'autoflow' or cfg.stage == 'depth_eigen' or cfg.stage == 'depth_sintel':   # seperate depth and flow training start from pretrained model all parts!!!
            model.load_state_dict(torch.load(cfg.restore_ckpt, map_location=gpu0), strict=True)  #Simply load all model parts!!!
        
        elif cfg.stage == 'joint_eigen' or cfg.stage == 'joint_sintel':  # joint training start from pretrained flow encoder and two seperate decoder heads!!!
            complete_state_dict1 = torch.load(cfg.restore_ckpt_flow, map_location=gpu0)
            # Load the filtered state dictionary into the model, with strict=False to allow for missing keys
            model.load_state_dict(complete_state_dict1, strict=False)
            
            complete_state_dict2 = torch.load(cfg.restore_ckpt_depth, map_location=gpu0)
            # Filter out keys starting from 'module.memory_decoder.update_block'
            filtered_state_dict2 = {k: v for k, v in complete_state_dict2.items() if k.startswith('module.depth_head')}
            # Load the filtered state dictionary into the model, with strict=False to allow for missing keys
            model.load_state_dict(filtered_state_dict2, strict=False)               
            
            
    model.cuda()
    model.train()

    
    train_loader = datasets.fetch_dataloader(cfg)
    optimizer, scheduler = fetch_optimizer(model, cfg.trainer)

    total_steps = 0
    scaler = GradScaler(enabled=cfg.mixed_precision)
    logger = Logger(model, scheduler, cfg)

    add_noise = False

    should_keep_training = True
    while should_keep_training:
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()      
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if cfg.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
            
            output = {}
            
            flow_predictions, depth_predictions = model(image1, image2, output)
            
            if cfg.type == 'flow':    
                loss, metrics = sequence_loss_flow(flow_predictions, flow, valid, cfg)  
            elif cfg.type == 'depth':
                loss, metrics = sequence_loss_depth(depth_predictions, flow, valid, cfg) 
            elif cfg.type == 'joint':
                loss1, metrics1 = sequence_loss_flow(flow_predictions, flow, valid, cfg)    
                loss2, metrics2 = sequence_loss_depth(depth_predictions, flow, valid, cfg)    
                loss = loss1 + loss2
                metrics = {**metrics1, **metrics2}            
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            metrics.update(output)
            logger.push(metrics)

            ### change evaluate to functions

            if total_steps % cfg.val_freq == cfg.val_freq - 1:
                PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps+1, cfg.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in cfg.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module, root=cfg.root))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module, root=cfg.root))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module, root=cfg.root))
                    elif val_dataset == 'depth_eigen':
                        results.update(evaluate.validate_depth_eigen(model.module, root=cfg.root, args=args))
                    elif val_dataset == 'depth_sintel':
                        results.update(evaluate.validate_depth_sintel(model.module, root=cfg.root, args=args))
                    elif val_dataset == 'depth_VKITTI_pretrain':
                        results.update(evaluate.validate_VKITTI_depth(model.module, root=cfg.root, args=args))
                
                logger.write_dict(results)
                model.train()
            
            total_steps += 1

            if total_steps > cfg.trainer.num_steps:
                should_keep_training = False
                break
            
    logger.close()
    PATH = cfg.log_dir + '/final'
    torch.save(model.state_dict(), PATH)

    PATH = f'checkpoints/{cfg.stage}.pth'
    torch.save(model.state_dict(), PATH)

    return PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='Protoformer', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
#    parser.add_argument('--local-rank', type=int, default=-1, metavar='N', help='Local process rank.') 
    parser.add_argument('--root', type=str)
    
    args = parser.parse_args()

    if args.stage == 'chairs':
        from configs.default import get_cfg
    elif args.stage == 'things':
        from configs.things import get_cfg
    elif args.stage == 'sintel':
        from configs.sintel import get_cfg
    elif args.stage == 'kitti':
        from configs.kitti import get_cfg
    elif args.stage == 'autoflow':
        from configs.autoflow import get_cfg
        
    elif args.stage == 'depth_eigen':
        from configs.depth_eigen import get_cfg
    elif args.stage == 'depth_sintel':
        from configs.depth_sintel import get_cfg
    elif args.stage == 'depth_VKITTI_pretrain':
        from configs.depth_VKITTI_pretrain import get_cfg        
        
    elif args.stage == 'joint_eigen':
        from configs.joint_eigen import get_cfg
    elif args.stage == 'joint_sintel':
        from configs.joint_sintel import get_cfg   
        
        

    cfg = get_cfg()
    cfg.update(vars(args))
    process_cfg(cfg)
    loguru_logger.add(str(Path(cfg.log_dir) / 'log.txt'), encoding="utf8")
    loguru_logger.info(cfg)

    torch.multiprocessing.set_start_method('spawn')
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(cfg)
