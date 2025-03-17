import numpy as np
import torch
from mmcv.transforms import BaseTransform, Compose
from embodiedbot.registry import TRANSFORMS

def depth2globalcoords(depths, cam2img, cam2global):
    """ 
        depths: (V, H, W)
        cam2img: (V, 4, 4) #intrinsics
        cam2global: (V, 4, 4)
        
    """
    V, H, W = depths.shape
    y = torch.arange(0, H).to(depths.device)
    x = torch.arange(0, W).to(depths.device)
    y, x = torch.meshgrid(y, x)

    x = x.unsqueeze(0).repeat(V, 1, 1).view(V, H*W)     # (V, H*W)
    y = y.unsqueeze(0).repeat(V, 1, 1).view(V, H*W)     # (V, H*W)

    fx = cam2img[:, 0, 0].unsqueeze(-1).repeat(1, H*W)
    fy = cam2img[:, 1, 1].unsqueeze(-1).repeat(1, H*W)
    cx = cam2img[:, 0, 2].unsqueeze(-1).repeat(1, H*W)
    cy = cam2img[:, 1, 2].unsqueeze(-1).repeat(1, H*W)

    z = depths.view(V, H*W)      # (V, H*W)
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    cam_coords = torch.stack([
        x, y, z, torch.ones_like(x)
    ], -1)      # (V, H*W, 4)

    world_coords = (cam2global @ cam_coords.permute(0, 2, 1)).permute(0, 2, 1)       # (V, H*W, 4)
    world_coords = world_coords[..., :3] / world_coords[..., 3].unsqueeze(-1)   # (V, H*W, 3)
    world_coords = world_coords.view(V, H, W, 3)

    return world_coords
@TRANSFORMS.register_module()
class ConvertDepthToCoordinate(BaseTransform):
    def __init__(self):
        super().__init__()
    def transform(self, results):
        depths = results['depth'] # (V, H, W)
        cam2global = results['cam2global'] # (V, 4, 4)
        cam2img =  results['cam2img'] # (4, 4)
        cam2img = cam2img.unsqueeze(0).repeat(depths.shape[0], 1, 1)
        axis_align_matrix = results['axis_align_matrix']
        axis_align_matrix = axis_align_matrix.unsqueeze(0).repeat(depths.shape[0], 1, 1) # (V, 4, 4)
        cam2align_global = torch.bmm(axis_align_matrix, cam2global)
        global_coords = depth2globalcoords(depths, cam2img, cam2align_global)
        results['global_coords'] = global_coords
        return results