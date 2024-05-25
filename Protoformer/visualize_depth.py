import sys
sys.path.append('core')
from PIL import Image
from glob import glob
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from configs.demo_depth import get_cfg  
from core.utils.misc import process_cfg
import datasets
from utils import flow_viz
from utils import frame_utils
import cv2
import math
import os.path as osp

cmap = plt.cm.viridis

from core.Protoformer import build_Protoformer
from utils.utils import InputPadder, forward_interpolate
import itertools
from skimage.transform import resize

TRAIN_SIZE = [368, 800] # adjust acording to settings


def flip_lr(image):
  
    assert image.dim() == 4, 'You need to provide a [B,C,H,W] image to flip'
    return torch.flip(image, [3])


def fuse_depth(depth, depth_hat, method='mean'):
    """
    Fuse depth and flipped depth maps

    """
    if method == 'mean':
        return 0.5 * (depth + depth_hat)
    elif method == 'max':
        return torch.max(depth, depth_hat)
    elif method == 'min':
        return torch.min(depth, depth_hat)
    else:
        raise ValueError('Unknown post-process method {}'.format(method))
    

def post_process_depth(depth, depth_flipped, method='mean'):
    """
    Post-process a depth map and flipped depth map  

    """
    
    B, C, H, W = depth.shape
    depth_hat = flip_lr(depth_flipped)
    depth_fused = fuse_depth(depth, depth_hat, method=method) 

    xs = torch.linspace(0., 1., W, device=depth.device,
                        dtype=depth.dtype).repeat(B, C, H, 1) 
    
    # # If real depth
    # mask = torch.clamp(20. * (xs - 0.95), 0., 1.)  
    # If inverse depth
    mask = 1.0 - torch.clamp(20. * (xs - 0.05), 0., 1.)
    
    mask_hat = flip_lr(mask)
    return mask_hat * depth + mask * depth_hat + \
           (1.0 - mask - mask_hat) * depth_fused



def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3



def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # HWC


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
  if min_overlap >= TRAIN_SIZE[0] or min_overlap >= TRAIN_SIZE[1]:
    raise ValueError(
        f"Overlap should be less than size of patch (got {min_overlap}"
        f"for patch size {patch_size}).")
  if image_shape[0] == TRAIN_SIZE[0]:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0]))
  else:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0] - min_overlap))
  if image_shape[1] == TRAIN_SIZE[1]:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1]))
  else:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1] - min_overlap))

  # Make sure the final patch is flush with the image boundary
  hs[-1] = image_shape[0] - patch_size[0]
  ws[-1] = image_shape[1] - patch_size[1]
  return [(h, w) for h in hs for w in ws]

def compute_weight(hws, image_shape, patch_size=TRAIN_SIZE, sigma=1.0, wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5 
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h+patch_size[0], w:w+patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx+1, h:h+patch_size[0], w:w+patch_size[1]])

    return patch_weights



def compute_depth(model, image1, image2, weights=None):
    print(f"computing depth map...")


    image_size = image1.shape[1:]

    image1, image2 = image1[None].cuda(), image2[None].cuda()

    hws = compute_grid_indices(image_size)
    if weights is None:    
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        _, flow_pre = model(image1, image2)

        flow_pre = padder.unpad(flow_pre)
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

    return flow


def compute_adaptive_image_size(image_size):
    target_size = TRAIN_SIZE
    scale0 = target_size[0] / image_size[0]
    scale1 = target_size[1] / image_size[1] 

    if scale0 > scale1:
        scale = scale0
    else:
        scale = scale1

    image_size = (int(image_size[1] * scale), int(image_size[0] * scale))

    return image_size

def prepare_image(root_dir, viz_root_dir, fn1, fn2, keep_size):
    print(f"preparing image...")
    print(f"root dir = {root_dir}, fn = {fn1}")

    image1 = frame_utils.read_gen(osp.join(root_dir, fn1))
    image2 = frame_utils.read_gen(osp.join(root_dir, fn2))
    image1 = np.array(image1).astype(np.uint8)[..., :3]
    image2 = np.array(image2).astype(np.uint8)[..., :3]
    if not keep_size:
        dsize = compute_adaptive_image_size(image1.shape[0:2])
        image1 = cv2.resize(image1, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        image2 = cv2.resize(image2, dsize=dsize, interpolation=cv2.INTER_CUBIC)
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()


    dirname = osp.dirname(fn1)
    filename = osp.splitext(osp.basename(fn1))[0]

    viz_dir = osp.join(viz_root_dir, dirname)
    if not osp.exists(viz_dir):
        os.makedirs(viz_dir)

    viz_fn = osp.join(viz_dir, filename + '.png')

    return image1, image2, viz_fn

def build_model():
    print(f"building  model...")
    cfg = get_cfg()
    model = torch.nn.DataParallel(build_Protoformer(cfg))
    model.load_state_dict(torch.load(cfg.model, map_location = torch.device('cuda:0')))

    model.cuda()
    model.eval()

    return model


def visualize_dep(root_dir, viz_root_dir, model, img_pairs, gt, keep_size):       

    weights = None 
    errors = []
    
    for img_pair, gt in zip(img_pairs, gt):
        fn1, fn2 = img_pair
        print(f"processing {fn1}, {fn2}...")

        image1, image2, viz_fn = prepare_image(root_dir, viz_root_dir, fn1, fn2, keep_size)

        depth = compute_depth(model, image1, image2, weights)  #### the depth here is actually disp or inv_depth!  because prediction converts to inv_depth during training

        image1_flipped = flip_lr(image1) 
        image2_flipped = flip_lr(image2)
        
        image1_flipped = image1_flipped.squeeze(0)
        image2_flipped = image2_flipped.squeeze(0)

        depth_flipped = compute_depth(model, image1_flipped, image2_flipped, weights) 

        depth =  np.expand_dims(depth, axis=0)
        depth_flipped = np.expand_dims(depth_flipped, axis=0)
        depth = depth.transpose(0, 3, 1, 2)
        depth_flipped = depth_flipped.transpose(0, 3, 1, 2)
        
        depth = torch.from_numpy(depth)
        depth_flipped = torch.from_numpy(depth_flipped)
        
        depth = post_process_depth(depth, depth_flipped)
        
        depth = depth.squeeze(0)
        depth = depth.permute(1,2,0)
        depth = depth.numpy()

        d_min = np.min(depth)
        d_max = np.max(depth)
        depth_relative = (depth - d_min) / (d_max - d_min)
        final_depth = 255.*depth_relative
        cv2.imwrite(viz_fn, final_depth.astype('uint8'))           



def process_sintel(sintel_dir):            
    img_pairs = []
    gt = []
    for scene in os.listdir(sintel_dir):
        dirname = osp.join(sintel_dir, scene, 'image_02/data')
        image_list = sorted(glob(osp.join(dirname, '*.jpg')))
        dirname_gt = osp.join(sintel_dir, scene, 'processed_depth/groundtruth/image_02')  ##############################
        gt_list = sorted(glob(osp.join(dirname_gt, '*.png')))        
        
        for i in range(len(image_list)-1):
            img_pairs.append((image_list[i], image_list[i+1]))
            gt.append(gt_list[i])

    return img_pairs, gt


def generate_pairs(dirname, start_idx, end_idx):
    img_pairs = []
    for idx in range(start_idx, end_idx):
        img1 = osp.join(dirname, f'{idx:06}.png')
        img2 = osp.join(dirname, f'{idx+1:06}.png')
        # img1 = f'{idx:06}.png'
        # img2 = f'{idx+1:06}.png'
        img_pairs.append((img1, img2))

    return img_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_type', default='sintel')
    parser.add_argument('--root_dir', default='.')
    parser.add_argument('--sintel_dir', default='datasets/KITTI_Eigen_depth/test') 
    parser.add_argument('--seq_dir', default='output_depth/final')
    parser.add_argument('--start_idx', type=int, default=1)     # starting index of the image sequence
    parser.add_argument('--end_idx', type=int, default=500)    # ending index of the image sequence
    parser.add_argument('--viz_root_dir', default='output_depth')
    parser.add_argument('--keep_size', action='store_true')     # keep the image size, or the image will be adaptively resized.)
    
    
    args = parser.parse_args()

    root_dir = args.root_dir
    viz_root_dir = args.viz_root_dir

    model = build_model()

    if args.eval_type == 'sintel':
        img_pairs, gt = process_sintel(args.sintel_dir)
    elif args.eval_type == 'seq':
        img_pairs = generate_pairs(args.seq_dir, args.start_idx, args.end_idx)
    with torch.no_grad():
        visualize_dep(root_dir, viz_root_dir, model, img_pairs, gt, args.keep_size)