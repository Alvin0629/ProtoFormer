from re import S
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import os    
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"    
import imageio
import cv2
import os
import math
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor, SparseDepthAugmentor, SparseDepthAugmentor_KITTI
from PIL import Image
import PIL.Image as pil

import skimage
import scipy
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from PIL import Image
import numba 

import logging
logger = logging.getLogger(__name__)

TAG_FLOAT = 202021.25

def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, vkitti2=False):
        self.augmentor = None
        self.sparse = sparse

        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.vkitti2 = vkitti2

    def __getitem__(self, index):
        #print(self.flow_list[index])
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0], test=self.is_test)
            img2 = frame_utils.read_gen(self.image_list[index][1], test=self.is_test)
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            if self.vkitti2:
                flow, valid = frame_utils.read_vkitti2_flow(self.flow_list[index])
            else:
                flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])  # [H, W, 2], [H, W]
       
        else:
            flow = frame_utils.read_gen(self.flow_list[index])
            

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        
        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        return img1, img2, flow, valid.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)



class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)
        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


#Git: 'datasets/FlyingChairs_release'
class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release'):
        super(FlyingChairs, self).__init__(aug_params)
        
        images = sorted(glob(osp.join(root+'/data', '*.ppm')))
        flows = sorted(glob(osp.join(root+'/data', '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('FlyingChairs_train_val.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


#Git: 'datasets/FlyingThings3D'
class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass', split='training'):
        super(FlyingThings3D, self).__init__(aug_params)

        split_dir = 'TRAIN' if split == 'training' else 'TEST'
        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, f'{split_dir}/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, f'optical_flow/{split_dir}/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
      

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
                
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))
            
class KITTI12(FlowDataset):
    def __init__(self, aug_params=None, split='training',
                 root='datasets/KITTI12'):
        super(KITTI12, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'colored_0/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'colored_0/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = os.path.basename(img1)
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class VKITTI2(FlowDataset):
    def __init__(self, aug_params=None,
                 root='datasets/VKITTI2',
                 ):
        super(VKITTI2, self).__init__(aug_params, sparse=True, vkitti2=True,
                                      )

        scenes = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']

        for scene in scenes:
            scene_dir = os.path.join(root, scene)

            types = os.listdir(scene_dir)

            for scene_type in types:
                type_dir = os.path.join(scene_dir, scene_type)

                imgs = sorted(glob(os.path.join(type_dir, 'frames', 'rgb', 'Camera_0', '*.jpg')))

                flows_fwd = sorted(glob(os.path.join(type_dir, 'frames', 'forwardFlow', 'Camera_0', '*.png')))
                flows_bwd = sorted(glob(os.path.join(type_dir, 'frames', 'backwardFlow', 'Camera_0', '*.png')))

                assert len(imgs) == len(flows_fwd) + 1 and len(imgs) == len(flows_bwd) + 1

                for i in range(len(imgs) - 1):
                    # forward
                    self.image_list += [[imgs[i], imgs[i + 1]]]
                    self.flow_list += [flows_fwd[i]]

                    # backward
                    self.image_list += [[imgs[i + 1], imgs[i]]]
                    self.flow_list += [flows_bwd[i]]



class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)
        
        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1


class Eigen_Depth(data.Dataset):       
    def __init__(self, aug_params=None, sparse=True, split='training',
                 root='datasets/KITTI_Eigen_depth',
                 ):

        self.aug_params = aug_params
        self.sparse = sparse
        if self.aug_params is not None:
            if sparse:
                self.augmentor = SparseDepthAugmentor(**aug_params) 


        self.flow_list = []
        self.image_list = []

        if split == 'training':
            self.data_path = 'datasets/KITTI_Eigen_depth/train'
            self.gt_path =  'datasets/KITTI_Eigen_depth/train'
        elif split == 'val':
            self.data_path = 'datasets/KITTI_Eigen_depth/val'
            self.gt_path =  'datasets/KITTI_Eigen_depth/val'
            

        types_imgs = os.listdir(self.data_path)
        types_depths = os.listdir(self.gt_path)

        for scene_type in types_depths:

            imgs = sorted(glob(os.path.join(self.data_path, scene_type, 'image_02', 'data', '*.png')) + glob(os.path.join(self.data_path, scene_type, 'image_02', 'data', '*.jpg')))
            flows_fwd = sorted(glob(os.path.join(self.gt_path, scene_type, 'processed_depth', 'groundtruth', 'image_02', '*.png')) + glob(os.path.join(self.gt_path, scene_type, 'processed_depth', 'groundtruth', 'image_02', '*.jpg')))
            

            for i in range(len(imgs) - 1):
                # forward
                self.image_list += [[imgs[i], imgs[i + 1]]]
                self.flow_list += [flows_fwd[i]]


    def augment_image(self, img1, img2):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image1_aug = img1 ** gamma
        image2_aug = img2 ** gamma
        
        # brightness augmentation
        brightness = random.uniform(0.9, 1.1)
        image1_aug = image1_aug * brightness
        image2_aug = image2_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image1_aug.shape[0], image1_aug.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image1_aug *= color_image
        image1_aug = np.clip(image1_aug, 0, 1)
        
        image2_aug *= color_image
        image2_aug = np.clip(image2_aug, 0, 1)

        return image1_aug, image2_aug



        
    def __getitem__(self, index):   
        
        img1 = Image.open(self.image_list[index][0])
        img2 = Image.open(self.image_list[index][1])

        depth_gt = pil.open(self.flow_list[index]) 
        
        height = img1.height
        width = img1.width
        
        img1 = np.array(img1).astype(np.uint8) 
        img2 = np.array(img2).astype(np.uint8)  
        
        
        depth_gt = depth_gt.resize((width, height), pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32)
        
        
        ## do augment on gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            img1, img2 = self.augment_image(img1, img2)


        if self.aug_params is not None:
            if self.sparse:
                img1, img2, depth_gt = self.augmentor(img1, img2, depth_gt)


        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        
        if len(depth_gt.shape) == 2:    
            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_gt = torch.from_numpy(depth_gt).permute(2, 0, 1).float()
        else:
            depth_gt = torch.from_numpy(depth_gt).permute(2, 0, 1).float()
            
            
        depth_gt = depth_gt.clamp(0.1, 80) 
        valid = (depth_gt[0].abs() <= 80) 

        return img1, img2, depth_gt, valid.float()            

    
    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self        
        
    def __len__(self):
        return len(self.image_list)



class VKITTI_Depth(data.Dataset):   #VKITTI
    def __init__(self, aug_params=None, sparse=True, split='training',
                 root='datasets/VKITTI_depth/vkitti_depth',
                 ):
        self.aug_params = aug_params
        self.sparse = sparse
        if self.aug_params is not None:
            if sparse:
                self.augmentor = SparseDepthAugmentor(**aug_params)

        self.flow_list = []
        self.image_list = []


        self.data_path = 'datasets/VKITTI_depth/vkitti_depth'
        self.gt_path =  'datasets/VKITTI_depth/vkitti_depth'

        scenes = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']

        for scene in scenes:
            scene_dir = os.path.join(self.data_path, scene)


            types_imgs = os.listdir(scene_dir)
            types_depths = os.listdir(scene_dir)

            for scene_type in types_depths:

                imgs = sorted(glob(os.path.join(self.data_path, scene,  scene_type, 'frames', 'rgb', 'Camera_0', '*.jpg')))
                flows_fwd = sorted(glob(os.path.join(self.gt_path, scene, scene_type, 'frames', 'depth', 'Camera_0', '*.png')))

                for i in range(len(imgs) - 1):
                    # forward
                    self.image_list += [[imgs[i], imgs[i + 1]]]
                    self.flow_list += [flows_fwd[i]]


    def augment_image(self, img1, img2):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image1_aug = img1 ** gamma
        image2_aug = img2 ** gamma
        
        # brightness augmentation
        brightness = random.uniform(0.9, 1.1)
        image1_aug = image1_aug * brightness
        image2_aug = image2_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image1_aug.shape[0], image1_aug.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image1_aug *= color_image
        image1_aug = np.clip(image1_aug, 0, 1)
        
        image2_aug *= color_image
        image2_aug = np.clip(image2_aug, 0, 1)

        return image1_aug, image2_aug

    
    def __getitem__(self, index):
        img1 = Image.open(self.image_list[index][0])
        img2 = Image.open(self.image_list[index][1])


        depth_gt = pil.open(self.flow_list[index])
        depth_gt = np.array(depth_gt).astype(np.float32) * 80.0 / 65535.0

        img1 = np.array(img1).astype(np.uint8) 
        img2 = np.array(img2).astype(np.uint8) 


        # do augment on gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            img1, img2 = self.augment_image(img1, img2)


        if self.aug_params is not None:
            if self.sparse:
                img1, img2, depth_gt = self.augmentor(img1, img2, depth_gt)


        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        if len(depth_gt.shape) == 2:   
            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_gt = torch.from_numpy(depth_gt).permute(2, 0, 1).float()
        else:
            depth_gt = torch.from_numpy(depth_gt).permute(2, 0, 1).float()

        
        depth_gt = depth_gt.clamp(0.1, 80) 

        valid = (depth_gt[0].abs() <= 80) 

        return img1, img2, depth_gt, valid.float()            
    
    
    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self            
    
    def __len__(self):
        return len(self.image_list)





class Sintel_Depth(data.Dataset):        
    def __init__(self, aug_params=None, sparse=True, split='training',
                 root='MPI-Sintel-depth', dstype='clean'):
        self.aug_params = aug_params
        self.sparse = sparse
        if self.aug_params is not None:
            if sparse:
                self.augmentor = SparseDepthAugmentor(**aug_params)


        self.flow_list = []
        self.image_list = []

        self.data_path = 'datasets/MPI-Sintel-depth/train'
        self.gt_path =  'datasets/MPI-Sintel-depth/train'


        types_imgs = os.listdir(self.data_path)
        types_depths = os.listdir(self.gt_path)

        for scene_type in types_depths:
            
            imgs = sorted(glob(os.path.join(self.data_path, scene_type, dstype, '*.png')))

            flows_fwd = sorted(glob(os.path.join(self.gt_path, scene_type, 'depth', '*.dpt')))
            
            for i in range(len(imgs) - 1):
                # forward
                self.image_list += [[imgs[i], imgs[i + 1]]]
                self.flow_list += [flows_fwd[i]]


    def augment_image(self, img1, img2):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image1_aug = img1 ** gamma
        image2_aug = img2 ** gamma
        
        # brightness augmentation
        brightness = random.uniform(0.9, 1.1)
        image1_aug = image1_aug * brightness
        image2_aug = image2_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image1_aug.shape[0], image1_aug.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image1_aug *= color_image
        image1_aug = np.clip(image1_aug, 0, 1)
        
        image2_aug *= color_image
        image2_aug = np.clip(image2_aug, 0, 1)

        return image1_aug, image2_aug

    
        
    def __getitem__(self, index):
        img1 = Image.open(self.image_list[index][0])
        img2 = Image.open(self.image_list[index][1])


        depth_gt = depth_read(self.flow_list[index])     

        depth_gt = np.resize(depth_gt, (436, 1024))       

        
        img1 = np.array(img1).astype(np.uint8) 
        img2 = np.array(img2).astype(np.uint8) 


        depth_gt = np.expand_dims(depth_gt, axis=2)
        depth_gt = depth_gt  


        # ## do augment on gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            img1, img2 = self.augment_image(img1, img2)


        if self.aug_params is not None:
            if self.sparse:
                img1, img2, depth_gt = self.augmentor(img1, img2, depth_gt)


        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        
        
        if len(depth_gt.shape) == 2:   
            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_gt = torch.from_numpy(depth_gt).permute(2, 0, 1).float()
        else:
            depth_gt = torch.from_numpy(depth_gt).permute(2, 0, 1).float()


        depth_gt = depth_gt.clamp(0.1, 80) 
        valid = (depth_gt[0].abs() <= 80) 

        return img1, img2, depth_gt, valid.float()            

    
    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self        
        
    def __len__(self):
        return len(self.image_list)






def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """


    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training',root=args.root)
    
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, root="/FlyingThings3D", dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, root="/FlyingThings3D", dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, root="/FlyingThings3D", dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, root="/Sintel", split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, root="/Sintel", split='training', dstype='final')

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3,
                           'max_scale': 0.5, 'do_flip': True}, root="/KITTI")
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True},
                        root="/HD1K")
            vkitti2 = VKITTI2({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})  # 42420
            
            train_dataset = 100*sintel_clean + 120*sintel_final + 200*kitti + 5*hd1k + things #+ vkitti2

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 120*sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, root="/KITTI",split='training')

    elif args.stage == 'sintel_ft':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}

        sintel_clean = MpiSintel(aug_params, root="/Sintel", split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, root="/Sintel", split='training', dstype='final')

        train_dataset = sintel_clean + sintel_final



    elif args.stage == 'depth_eigen':
        print('Correct! using Eigen for depth estimation train')
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = Eigen_Depth(aug_params, split='training')


    elif args.stage == 'depth_sintel':
        print('Correct! using Sintel for depth estimation train')
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': False}
        
        sintel_clean = Sintel_Depth(aug_params, split='training', dstype='clean')
        
        sintel_final = Sintel_Depth(aug_params, split='training', dstype='final')   
        
        train_dataset = sintel_clean + sintel_final


    elif args.stage == 'depth_VKITTI_pretrain':
        print('Correct! using VKITTI for depth estimation pretrain')
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': False}
        
        vkitti_depth = VKITTI_Depth(aug_params, split='training')
    
        train_dataset = vkitti_depth
        

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=True, shuffle=True, num_workers=4, drop_last=True)


    print('Training with %d image pairs' % len(train_dataset))
    return train_loader
