import numpy as np
import torch
from mmcv.transforms import BaseTransform, Compose
from embodiedbot.registry import TRANSFORMS

@TRANSFORMS.register_module()
class PackBotData(BaseTransform):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys
    def transform(self, results):
        packed_results = {}
        for key in self.keys:
            packed_results[key] = results.get(key,None)
        return packed_results