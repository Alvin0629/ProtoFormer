import torch
def build_Protoformer(cfg):
    name = cfg.transformer 
    if name == 'latentcostformer':
        from .LatentCostFormer.transformer import Protoformer
    else:
        raise ValueError(f"Not a valid architecture!")

    return Protoformer(cfg[name])