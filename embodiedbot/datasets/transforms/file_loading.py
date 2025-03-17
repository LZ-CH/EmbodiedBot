from typing import Optional

import mmcv
import mmengine
import numpy as np
from mmcv.transforms import BaseTransform
from embodiedbot.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadDepthFromFile(BaseTransform):
    """Load a depth image from file.

    Required Keys:

    - depth_path

    Modified Keys:

    - depth
    - depth_shape

    Args:
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 imdecode_backend: str = 'cv2',
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.imdecode_backend = imdecode_backend

        self.backend_args = None
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        filename = results['depth_path']
        depth_scale = results['depth_scale']
        depth_offset = results.get('depth_offset', 0)
        try:
            depth_bytes = mmengine.fileio.get(
                filename, backend_args=self.backend_args)
            depth = mmcv.imfrombytes(depth_bytes,
                                         flag='unchanged',
                                         backend=self.imdecode_backend).astype(
                                             np.float32) / depth_scale
            depth = depth+depth_offset
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e

        results['depth'] = depth
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.backend_args is not None:
            repr_str += f'backend_args={self.backend_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str