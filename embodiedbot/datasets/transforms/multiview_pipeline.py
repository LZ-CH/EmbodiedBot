import numpy as np
import torch
from mmcv.transforms import BaseTransform, Compose

from embodiedbot.registry import TRANSFORMS
import numpy as np
@TRANSFORMS.register_module()
class MultiViewPipeline(BaseTransform):
    """Multiview data processing pipeline.

    The transform steps are as follows:

        1. Select frames.
        2. Re-ororganize the selected data structure.
        3. Apply transforms for each selected frame.
        4. Concatenate data to form a batch.

    Args:
        transforms (list[dict | callable]):
            The transforms to be applied to each select frame.
        n_images (int): Number of frames selected per scene.
        ordered (bool): Whether to put these frames in order.
            Defaults to True.
    """

    def __init__(self, transforms, n_images, sampling_strategy='uniform', ordered=True):
        super().__init__()
        self.transforms = Compose(transforms)
        self.n_images = n_images
        self.ordered = ordered
        self.sampling_strategy = sampling_strategy
        assert self.sampling_strategy in ['uniform', 'random']
    def transform(self, results: dict) -> dict:
        """Transform function.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: output dict after transformation.
        """
        imgs = []
        img_paths = []
        depth = []
        depth_paths = []
        cam2global = []
        total_num = len(results['img_path']) 
        if self.sampling_strategy == 'uniform':
            sample_idx = np.linspace(0, total_num-1, self.n_images, dtype=int)
            sample_idx = sample_idx.tolist()
        elif self.sampling_strategy == 'random':
            sample_idx = np.random.choice(total_num, self.n_images, replace=total_num<self.n_images)
            sample_idx = sample_idx.tolist()
        if self.ordered:
            sample_idx.sort()
        for i in sample_idx:
            cam2global.append(results['cam2global'][i])
            _results = dict()
            _results['img_path'] = results['img_path'][i]
            img_paths.append(_results['img_path'])
            if 'depth_path' in results:
                _results['depth_path'] = results['depth_path'][i]
                _results['depth_scale'] = results.get('depth_scale', 1)
                _results['depth_offset'] = results.get('depth_offset', 0)
                depth_paths.append(_results['depth_path'])
            _results = self.transforms(_results)
            imgs.append(_results['img'])
            if 'depth' in _results:
                depth.append(_results['depth'])
        results['img'] = torch.from_numpy(np.array(imgs))# (V, H, W, C)
        results['img_path'] = img_paths
        if len(depth):
            results['cam2global'] = torch.from_numpy(np.array(cam2global)).float() #(V, 4, 4)
            results['axis_align_matrix']=torch.from_numpy(results['axis_align_matrix']).float() #(4, 4)
            results['cam2img'] = torch.from_numpy(results['cam2img']).float() #(4, 4)
            results['depth'] = torch.from_numpy(np.array(depth)).float() #(V, H, W)
            # resize to img.shape
            results['depth'] = torch.nn.functional.interpolate(results['depth'], size=(results['img'].shape[1], results['img'].shape[2]))
            results['depth_path'] = depth_paths
        return results