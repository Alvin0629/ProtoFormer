
import torch.nn as nn
import numpy as np
import timm
from collections import OrderedDict

class twins_svt_encode(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.svt = twins_svt_large(pretrained=pretrained)
        del self.svt.head
        del self.svt.patch_embeds[3]
        del self.svt.patch_embeds[2]
        del self.svt.blocks[3]
        del self.svt.blocks[2]
        del self.svt.pos_block[3]
        del self.svt.pos_block[2]
        del self.svt.depths[3]
        

    def forward(self, x, data=None, layer=2):
        features=[]
        B = x.shape[0]
        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):

            x, size = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j==0:
                    x = pos_blk(x, size)
            if i < len(self.svt.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()

            # if i == layer-1:
            #     break
            features.append(x)     ## return [x1, x2]

        return features 



    def compute_params(self, layer=2):
        num = 0
        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(self.svt.patch_embeds, self.svt.pos_drops, self.svt.blocks, self.svt.pos_block)):

            for param in embed.parameters():
                num +=  np.prod(param.size())

            for param in drop.parameters():
                num +=  np.prod(param.size())

            for param in blocks.parameters():
                num +=  np.prod(param.size())

            for param in pos_blk.parameters():
                num +=  np.prod(param.size())

            if i == layer-1:
                break

        for param in self.svt.head.parameters():
            num +=  np.prod(param.size())

        return num



import math
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import Mlp, DropPath, to_2tuple, trunc_normal_
import os

__all__ = ['Twins']  # model_registry will add each entrypoint fn to this

Size_ = Tuple[int, int]
_EXPORTABLE = False
_HAS_FUSED_ATTN = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
if 'TIMM_FUSED_ATTN' in os.environ:
    _USE_FUSED_ATTN = int(os.environ['TIMM_FUSED_ATTN'])
else:
    _USE_FUSED_ATTN = 1  # 0 == off, 1 == on (for tested use), 2 == on (for experimental use)
def use_fused_attn(experimental: bool = False) -> bool:
    # NOTE: ONNX export cannot handle F.scaled_dot_product_attention as of pytorch 2.0
    if not _HAS_FUSED_ATTN or _EXPORTABLE:
        return False
    if experimental:
        return _USE_FUSED_ATTN > 1
    return _USE_FUSED_ATTN > 0

class LocallyGroupedAttn(nn.Module):
    """ LSA: self attention within a group
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., ws=1):
        assert ws != 1
        super(LocallyGroupedAttn, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, size: Size_):
        # There are two implementations for this function, zero padding or mask. We don't observe obvious difference for
        # both. You can choose any one, we recommend forward_padding because it's neat. However,
        # the masking implementation is more reasonable and accurate.
        B, N, C = x.shape
        H, W = size
        #print("input local shape: ", x.shape)
        x = x.view(B, H, W, C).contiguous()

        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3).contiguous()
        qkv = self.qkv(x).reshape(
            B, _h * _w, self.ws * self.ws, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5).contiguous()
        q, k, v = qkv.unbind(0)
        #print("local ground q,k,v shape: ", q.shape, k.shape, v.shape)
        # step1: [2, 252, 4, 49, 32] X X
        # step2: [2, 63, 8, 49, 32] X X
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1).contiguous()
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(2, 3).contiguous().reshape(B, _h, _w, self.ws, self.ws, C)
        x = x.transpose(2, 3).contiguous().reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        #print("output local shape: ", x.shape)
        #print(x.shape) [B,N,D] [2, 11408, 128] [2, 2852, 256])
        return x


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1).contiguous())
    return sim


class Clustering(nn.Module):             
    def __init__(self, dim, out_dim, center_w=2, center_h=2, window_w=2, window_h=2, heads=4, head_dim=24,
                 return_center=False, num_clustering=1):
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.conv1 = nn.Conv2d(dim, heads * head_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1)
        self.conv_c = nn.Conv2d(head_dim, head_dim, kernel_size=1)
        self.conv_v = nn.Conv2d(dim, heads * head_dim, kernel_size=1)
        self.conv_f = nn.Conv2d(dim, heads * head_dim, kernel_size=1)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((center_w, center_h))
        self.window_w = int(window_w)
        self.window_h = int(window_h)
        self.return_center = return_center
        self.softmax = nn.Softmax(dim=-2)
        self.num_clustering = num_clustering

    def forward(self, x, size):  # [b,c,w,h]
        B, N, C = x.shape
        H, W = size
        x = x.view(B, H, W, C).permute(0,3,1,2).contiguous()
        #print("input shape: ",x.shape)
        value = self.conv_v(x)
        feature = self.conv_f(x)
        x = self.conv1(x)

        # multi-head
        b, c, w, h = x.shape
        x = x.reshape(b * self.heads, int(c / self.heads), w, h)
        value = value.reshape(b * self.heads, int(c / self.heads), w, h)
        feature = feature.reshape(b * self.heads, int(c / self.heads), w, h)

        # window token
        if self.window_w > 1 and self.window_h > 1:
            b, c, w, h = x.shape
            x = x.reshape(b * self.window_w * self.window_h, c, int(w / self.window_w), int(h / self.window_h))
            value = value.reshape(b * self.window_w * self.window_h, c, int(w / self.window_w), int(h / self.window_h))
            feature = feature.reshape(b * self.window_w * self.window_h, c, int(w / self.window_w),
                                      int(h / self.window_h))

        b, c, w, h = x.shape
        value = value.reshape(b, w * h, c)

        # centers
        centers = self.centers_proposal(x)
        b, c, c_w, c_h = centers.shape
        centers_feature = self.centers_proposal(feature).reshape(b, c_w * c_h, c)

        feature = feature.reshape(b, w * h, c)

        for _ in range(self.num_clustering):  # iterative clustering and updating centers
            centers = self.conv_c(centers).reshape(b, c_w * c_h, c)
            similarity = self.softmax((F.normalize(centers,dim=-1) @ F.normalize(value,dim=-1).transpose(-2, -1).contiguous()))
            centers = (similarity @ feature).reshape(b, c, c_w, c_h)

        # similarity
        similarity = torch.sigmoid(
            self.sim_beta + self.sim_alpha * pairwise_cos_sim(centers.reshape(b, c, -1).permute(0, 2, 1).contiguous(),
                                                              x.reshape(b, c, -1).permute(0, 2, 1).contiguous()))

        # assign each point to one center
        _, max_idx = similarity.max(dim=1, keepdim=True)
        mask = torch.zeros_like(similarity)
        mask.scatter_(1, max_idx, 1.)
        similarity = similarity * mask

        out = ((feature.unsqueeze(dim=1) * similarity.unsqueeze(dim=-1)).sum(dim=2) + centers_feature) / (
                    mask.sum(dim=-1, keepdim=True) + 1.0)

        if self.return_center:
            out = out.reshape(b, c, c_w, c_h)
            return out
        else:
            out = (out.unsqueeze(dim=2) * similarity.unsqueeze(dim=-1)).sum(dim=1)
            out = out.reshape(b, c, w, h)

        # recover feature maps
        if self.window_w > 1 and self.window_h > 1:
            out = out.reshape(int(out.shape[0] / self.window_w / self.window_h), out.shape[1],
                              out.shape[2] * self.window_w, out.shape[3] * self.window_h)

        out = out.reshape(int(out.shape[0] / self.heads), out.shape[1] * self.heads, out.shape[2], out.shape[3])
        out = self.conv2(out)
        #print("out shape:", out.shape)
        out = out.reshape(B,C,W*H).permute(0,2,1).contiguous()
        #print("out shape:", out.shape)
        return out

class GlobalSubSampleAttn(nn.Module):
    """ GSA: using a  key to summarize the information for a group to be efficient.
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., sr_ratio=1,clustering=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.clustering = clustering,
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, size: Size_):
        #print("input global shape: ",x.shape)
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if self.sr is not None:
            x = x.permute(0, 2, 1).contiguous().reshape(B, C, *size)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x = self.norm(x)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv.unbind(0)
        #print("global q,k,v shape: ", q.shape, k.shape, v.shape)
        # step 1: [2, 4, 11408, 32][2, 4, 165, 32][2, 4, 165, 32]
        # step 2: [2, 8, 2852, 32][2, 8, 165, 32][2, 8, 165, 32]
        if self.fused_attn:
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1).contiguous()
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        #print("output global shape: ",x.shape)
        return x


class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            center_w, center_h,window_w,window_h,return_center,num_clustering,
            mlp_ratio=4.,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            sr_ratio=1,
            ws=1,
            adapter=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.attn = Clustering(dim=dim, out_dim=dim, center_w=center_w, center_h=center_h, window_w=window_w,
                                  window_h=window_h, heads=num_heads, head_dim=32, return_center=return_center,
                                  num_clustering=num_clustering)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.adapter_atn=nn.Sequential(OrderedDict([
            ("l1", nn.Linear(dim,dim//2)),
            ("relu",nn.ReLU()),
            ("l2",nn.Linear(dim//2, dim))]
        ))
        self.adapter_mlp = nn.Sequential(OrderedDict([
            ("l1", nn.Linear(dim, dim // 2)),
            ("relu", nn.ReLU()),
            ("l2", nn.Linear(dim // 2, dim))]
        ))
        self.adapter_atn.apply(self.init_adptr)
        self.adapter_mlp.apply(self.init_adptr)
        self.adptr=adapter
    def init_adptr(self, module):
        # reint adapter - small value + bias
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0,
                                       std=0.001)
            #print(module.weight.grad)
            if module.bias is not None:
                module.bias.data.zero_()
    def forward(self, x, size: Size_):
        if self.adptr:
            temp_x = self.attn(self.norm1(x), size)
            temp_x = temp_x + self.adapter_atn(temp_x)
            x=x+self.drop_path1(temp_x)
            temp_x = self.mlp(self.norm2(x))
            temp_x = temp_x + self.adapter_mlp(temp_x)
            x = x + self.drop_path2(temp_x)
        else:
            x = x + self.drop_path1(self.attn(self.norm1(x), size))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class PosConv(nn.Module):
    # PEG  from https://arxiv.org/abs/2102.10882
    def __init__(self, in_chans, embed_dim=768, stride=1):
        super(PosConv, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 3, stride, 1, bias=True, groups=embed_dim),
        )
        self.stride = stride

    def forward(self, x, size: Size_):
        B, N, C = x.shape
        cnn_feat_token = x.transpose(1, 2).view(B, C, *size).contiguous()
        x = self.proj(cnn_feat_token)
        if self.stride == 1:
            x += cnn_feat_token
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x) -> Tuple[torch.Tensor, Size_]:
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        out_size = (H // self.patch_size[0], W // self.patch_size[1])

        return x, out_size


class Twins(nn.Module):
    """ Twins Vision Transfomer (Revisiting Spatial Attention)

    Adapted from PVT (PyramidVisionTransformer) class at https://github.com/whai362/PVT.git
    """
    def __init__(
            self,
            img_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            global_pool='avg',
            embed_dims=(64, 128, 256, 512),
            num_heads=(1, 2, 4, 8),
            mlp_ratios=(4, 4, 4, 4),
            depths=(3, 4, 6, 3),
            sr_ratios=(8, 4, 2, 1),
            wss=None,
            drop_rate=0.,
            pos_drop_rate=0.,
            proj_drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            block_cls=Block,
            center_w=[10, 10, 10, 10], center_h=[10, 10, 10, 10], window_w=[0,0,0,0], window_h=[0,0,0,0],
            return_center=False, num_clustering=3,   ##num_clustering=1
            adapter=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.depths = depths
        self.embed_dims = embed_dims
        self.num_features = embed_dims[-1]
        self.grad_checkpointing = False

        img_size = to_2tuple(img_size)
        prev_chs = in_chans
        self.patch_embeds = nn.ModuleList()
        self.pos_drops = nn.ModuleList()
        for i in range(len(depths)):
            self.patch_embeds.append(PatchEmbed(img_size, patch_size, prev_chs, embed_dims[i]))
            self.pos_drops.append(nn.Dropout(p=pos_drop_rate))
            prev_chs = embed_dims[i]
            img_size = tuple(t // patch_size for t in img_size)
            patch_size = 2

        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for k in range(len(depths)):
            _block = nn.ModuleList([block_cls(
                dim=embed_dims[k],
                num_heads=num_heads[k],
                center_w=center_w[k],
                center_h=center_h[k],
                window_w=window_w[k],
                window_h=window_h[k],
                return_center=return_center,
                num_clustering=num_clustering,
                mlp_ratio=mlp_ratios[k],
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[k],
                ws=1 if wss is None or i % 2 == 1 else wss[k],
                adapter=adapter,
            ) for i in range(depths[k])],
            )
            self.blocks.append(_block)
            cur += depths[k]

        self.pos_block = nn.ModuleList([PosConv(embed_dim, embed_dim) for embed_dim in embed_dims])

        self.norm = norm_layer(self.num_features)

        # classification head
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set(['pos_block.' + n for n, p in self.pos_block.named_parameters()])

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^patch_embeds.0',  # stem and embed
            blocks=[
                (r'^(?:blocks|patch_embeds|pos_block)\.(\d+)', None),
                ('^norm', (99999,))
            ] if coarse else [
                (r'^blocks\.(\d+)\.(\d+)', None),
                (r'^(?:patch_embeds|pos_block)\.(\d+)', (0,)),
                (r'^norm', (99999,))
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        assert not enable, 'gradient checkpointing not supported'

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x):
        B = x.shape[0]
        for i, (embed, drop, blocks, pos_blk) in enumerate(
                zip(self.patch_embeds, self.pos_drops, self.blocks, self.pos_block)):
            x, size = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j == 0:
                    x = pos_blk(x, size)  # PEG here
            if i < len(self.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embeds.0.proj', 'classifier': 'head',
        **kwargs
    }

def twins_svt_large(pretrained=False, **kwargs) -> Twins:
    model_args = dict(
        patch_size=4, embed_dims=[128, 256, 512, 1024], num_heads=[4, 8, 16, 32], mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1])
    model = Twins(**dict(model_args, **kwargs))
    #if pretrained:
        # sct_weights = torch.load(pretrained, map_location='cpu')
        # model.load_state_dict(sct_weights)
    return model

if __name__ == "__main__":
    m = twins_svt_encode(pretrained=False)
    input = torch.randn(2, 3, 368, 496)
    out = m(input)
    print(out.shape)