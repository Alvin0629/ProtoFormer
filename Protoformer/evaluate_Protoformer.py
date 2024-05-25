import sys
sys.path.append('core')
from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from configs.default import get_cfg
from configs.things_eval import get_cfg as get_things_cfg
from core.utils.misc import process_cfg
import core.datasets as datasets
from utils import flow_viz
from utils import frame_utils

from core.Protoformer import build_Protoformer
from raft import RAFT
from utils.utils import InputPadder, forward_interpolate


@torch.no_grad()
def validate_chairs(model, root):
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation', root=root)
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        flow_pre, _ = model(image1, image2) 
        epe = torch.sum((flow_pre[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, root):
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        # if "Sintel" not in root:
        #     root += "/Sintel"
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype,root=root)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_pre, _ = model(image1, image2)   
            flow_pre = padder.unpad(flow_pre[0]).cpu()[0]
            
            epe = torch.sum((flow_pre - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results




@torch.no_grad()
def validate_depth_eigen(model, root, args):
    model.eval()
    val_dataset = datasets.Eigen_Depth(aug_params=None, split='val')

    d1_list, abs_list, sq_list = [], [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        _, flow_pre = model(image1, image2)  
        flow_pre = padder.unpad(flow_pre[1]).cpu()[0]    

        flow_pre  = 1. / flow_pre.clamp(min=0.0125) 
     
        scale = torch.median(flow_gt) / torch.median(flow_pre)
        flow_pre = flow_pre * scale
        
        thresh = torch.max((flow_gt / flow_pre), (flow_pre / flow_gt))
        d1 = (thresh < 1.25).float().mean()
        d2 = (thresh < 1.25 ** 2).float().mean()
        d3 = (thresh < 1.25 ** 3).float().mean()

        rms = (flow_gt - flow_pre) ** 2
        rms = torch.sqrt(rms.mean())

        log_rms = (torch.log(flow_gt) - torch.log(flow_pre)) ** 2
        log_rms = torch.sqrt(log_rms.mean())

        abs_rel = torch.mean(torch.abs(flow_gt - flow_pre) / flow_gt)
        sq_rel = torch.mean(((flow_gt - flow_pre) ** 2) / flow_gt)

        err = torch.log(flow_pre) - torch.log(flow_gt)
        silog = torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100.

        d1_list.append(d1.cpu().numpy())
        abs_list.append(abs_rel.cpu().numpy())
        sq_list.append(sq_rel.cpu().numpy())

    d1_list = np.array(d1_list)
    abs_list = np.array(abs_list)
    sq_list = np.array(sq_list)
    
    return_d1 = np.mean(d1_list)
    return_abs = np.mean(abs_list)
    return_sq = np.mean(sq_list)
    
    print("Validation Eigen depth: %f, %f, %f" % (return_d1, return_abs, return_sq))
    return {'Acc_d1': return_d1, 'Err_asbrel': return_abs, 'Err_sqrel': return_sq}



@torch.no_grad()
def validate_depth_sintel(model, root, args):
    model.eval()
    results = {}

    for dstype in ['clean', 'final']:
        val_dataset = datasets.Sintel_Depth(aug_params=None, split='training', dstype = dstype)

        d1_list, abs_list, sq_list = [], [], []
        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, valid_gt = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_pre = model(image1, image2)    

            flow_pre = padder.unpad(flow_pre[1]).cpu()[0]  
            
            flow_pre = 1. / flow_pre.clamp(min=0.0125)
        
            scale = torch.median(flow_gt) / torch.median(flow_pre)
            flow_pre = flow_pre * scale

            thresh = torch.maximum((flow_gt / flow_pre), (flow_pre / flow_gt))
            d1 = (thresh < 1.25).float().mean()
            d2 = (thresh < 1.25 ** 2).float().mean()
            d3 = (thresh < 1.25 ** 3).float().mean()

            rms = (flow_gt - flow_pre) ** 2
            rms = torch.sqrt(rms.mean())

            log_rms = (torch.log(flow_gt) - torch.log(flow_pre)) ** 2
            log_rms = torch.sqrt(log_rms.mean())

            abs_rel = torch.mean(torch.abs(flow_gt - flow_pre) / flow_gt)
            sq_rel = torch.mean(((flow_gt - flow_pre) ** 2) / flow_gt)

            err = torch.log(flow_pre) - torch.log(flow_gt)
            silog = torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100.

            d1_list.append(d1.cpu().numpy())
            abs_list.append(abs_rel.cpu().numpy())
            sq_list.append(sq_rel.cpu().numpy())

        d1_list = np.array(d1_list)
        abs_list = np.array(abs_list)
        sq_list = np.array(sq_list)
        
        return_d1 = np.mean(d1_list)
        return_abs = np.mean(abs_list)
        return_sq = np.mean(sq_list)
        
        print("Validation Sintel (%s) depth: %f, %f, %f" % (dstype, return_d1, return_abs, return_sq))
        results[dstype] = return_abs

    return results



@torch.no_grad()
def validate_VKITTI_depth(model, root, args):
    model.eval()
    results = {}

    val_dataset = datasets.VKITTI_Depth(aug_params=None, split='training')

    d1_list, abs_list, sq_list = [], [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        _, flow_pre = model(image1, image2)  

        flow_pre = padder.unpad(flow_pre[0]).cpu()[0] 

        flow_pre  = 1. / flow_pre.clamp(min=0.0125)  

        scale = torch.median(flow_gt) / torch.median(flow_pre)
        flow_pre = flow_pre * scale
        thresh = torch.maximum((flow_gt / flow_pre), (flow_pre / flow_gt))
        d1 = (thresh < 1.25).float().mean()
        d2 = (thresh < 1.25 ** 2).float().mean()
        d3 = (thresh < 1.25 ** 3).float().mean()

        rms = (flow_gt - flow_pre) ** 2
        rms = torch.sqrt(rms.mean())

        log_rms = (torch.log(flow_gt) - torch.log(flow_pre)) ** 2
        log_rms = torch.sqrt(log_rms.mean())

        abs_rel = torch.mean(torch.abs(flow_gt - flow_pre) / flow_gt)
        sq_rel = torch.mean(((flow_gt - flow_pre) ** 2) / flow_gt)

        err = torch.log(flow_pre) - torch.log(flow_gt)
        silog = torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100.

        d1_list.append(d1.cpu().numpy())
        abs_list.append(abs_rel.cpu().numpy())
        sq_list.append(sq_rel.cpu().numpy())
        
    d1_list = np.array(d1_list)
    abs_list = np.array(abs_list)
    sq_list = np.array(sq_list)

    return_d1 = np.mean(d1_list)
    return_abs = np.mean(abs_list)
    return_sq = np.mean(sq_list)
    
    print("Validation VKITTI Depth: %f, %f, %f" % (return_d1, return_abs, return_sq))
    return {'Acc_d1': return_d1, 'Err_asbrel': return_abs, 'Err_sqrel': return_sq}





@torch.no_grad()
def validate_kitti(model):
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_pre, _ = model(image1, image2)
        flow_pre = padder.unpad(flow_pre[0]).cpu()[0]

        epe = torch.sum((flow_pre - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()
        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset selection")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    cfg = get_things_cfg()
    
    cfg.update(vars(args))

    model = torch.nn.DataParallel(build_Protoformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    print(args)

    model.cuda()
    model.eval()

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            validate_sintel(model.module)

        elif args.dataset == 'kitti':
            validate_kitti(model.module)


