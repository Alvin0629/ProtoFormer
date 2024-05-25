
import torch
import torch.nn as nn
from ..encoders import twins_svt_encode
from .encoder import MemoryEncoder
from .decoder import flow_head, depth_head  
from .cnn import BasicEncoder

class Protoformer(nn.Module):
    def __init__(self, cfg):
        super(Protoformer, self).__init__()
        self.cfg = cfg

        self.memory_encoder = MemoryEncoder(cfg)
        self.flow_header = flow_head(cfg)
        self.depth_header = depth_head(cfg)     

        if cfg.cnet == 'twins':
            self.context_encoder = twins_svt_encode(pretrained=self.cfg.pretrain)
        elif cfg.cnet == 'basicencoder':
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn='instance')


    def forward(self, image1, image2, output=None, flow_init=None):
        # Following https://github.com/princeton-vl/RAFT/
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        data = {}

        if self.cfg.context_concat:
            context = self.context_encoder(torch.cat([image1, image2], dim=1))
        else:
            context = self.context_encoder(image1)
            
        cost_memory = self.memory_encoder(image1, image2, data, context[1])  
        
         ## Unified Protoformer training!
        if self.cfg.type == 'flow':
            flow_predictions = self.flow_header(cost_memory, context, data, flow_init=flow_init)
            depth_predictions = None
        elif self.cfg.type == 'depth':
            depth_predictions = self.depth_header(cost_memory, context, data, flow_init=flow_init)
            flow_predictions = None
        elif self.cfg.type == 'joint':
            flow_predictions = self.flow_header(cost_memory, context, data, flow_init=flow_init)
            depth_predictions = self.depth_header(cost_memory, context, data, flow_init=flow_init)
        
        return flow_predictions, depth_predictions
