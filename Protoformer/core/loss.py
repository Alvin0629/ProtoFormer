import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
MAX_FLOW = 400


def sequence_loss_flow(flow_preds, flow_gt, valid, cfg):
    """ Loss function defined over sequence of flow predictions """

    gamma = cfg.gamma
    max_flow = cfg.max_flow
    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    flow_gt_thresholds = [5, 10, 20]

    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()
    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        variance_focus = 0.85
        self.variance_focus = variance_focus
    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        dd = torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0
        return dd


def sequence_loss_depth(flow_preds, flow_gt, valid, cfg):
    gamma = cfg.gamma
    max_flow = cfg.max_flow
    n_predictions = len(flow_preds)
    flow_loss = 0.0
    flow_gt_thresholds = [5, 10, 20]
    variance_focus = 0.85
    depth_loss = silog_loss(variance_focus)
    mask = (flow_gt >= 0.1) & (flow_gt <= 80.0)  
    
    flow_preds_inv = flow_preds.copy()
    
    for i in range(n_predictions):
       
        flow_preds_inv[i]  = 1. / flow_preds_inv[i].clamp(min=0.0125)

        i_weight = gamma**(n_predictions - i - 1)        
        i_loss = depth_loss(flow_preds_inv[i], flow_gt, mask.to(torch.bool))   
        #gradient_loss = imgrad_loss(flow_preds_inv[i], flow_gt, mask.to(torch.bool))
        flow_loss += (i_weight *  i_loss).mean()
        #flow_loss += (0.1 * i_weight * gradient_loss).mean()
    scale = torch.median(flow_gt) / torch.median(flow_preds_inv[-1])
        
    flow_preds_inv[-1] = flow_preds_inv[-1] * scale
    thresh = torch.max((flow_gt / flow_preds_inv[-1]), (flow_preds_inv[-1] / flow_gt))
    d1 = (thresh < 1.25).float().mean()
    d2 = (thresh < 1.25**2).float().mean()
    d3 = (thresh < 1.25**3).float().mean()

    rms = (flow_gt - flow_preds_inv[-1]) ** 2
    rms = torch.sqrt(rms.mean())
    epsilon = 1e-6
    abs_rel = torch.mean(torch.abs(flow_gt - flow_preds_inv[-1]) / (flow_gt + epsilon))
    sq_rel = torch.mean(((flow_gt - flow_preds_inv[-1]) ** 2) / (flow_gt + epsilon))
    metrics = {
        'rms': rms.float().item(),
        'abs_rel': abs_rel.float().item(),
        'sq_rel': sq_rel.float().item(),
        'd1': d1.float().item(),
        'd2': d2.float().item(),
        'd3': d3.float().item(),
    }
    return flow_loss, metrics
