from mmengine.dataset import BaseDataset
import logging
from mmengine.fileio import join_path, load
from collections.abc import Mapping
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from mmengine.config import Config

class ScanNetTaskBaseDataset(BaseDataset):
    def __init__(self, 
                 ann_file: Optional[str] = '',
                 scannet_info_file: Optional[str] = '',
                 data_root: Optional[str] = '',
                 data_prefix: dict = dict(img_path='', depth_path=''),
                 metainfo: Union[Mapping, Config, None] = None,
                 test_mode: bool = False,
                 **kwargs):
        if metainfo is None:
            metainfo = dict(data_type='scannet_task_dataset', 
                            task_name='scannet_task',
                            )
        self.scannet_info_file = join_path(data_root, scannet_info_file) if scannet_info_file != '' else scannet_info_file
        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)
        
    def merge_scannet_info(self, data_list):
        if self.scannet_info_file == '':
            logging.warning('scannet info file is empty')
            return data_list
        scene_list = load(self.scannet_info_file)['data_list']
        scene_dict = {scene['sample_idx']: scene for scene in scene_list}
        for data in data_list:
            scene = scene_dict[data['video']]
            new_data_info = {}
            new_data_info['img_path'] = [img['img_path'] for img in scene['images']]
            new_data_info['depth_path'] = [img['depth_path'] for img in scene['images']]
            new_data_info['cam2global'] = [img['cam2global'] for img in scene['images']] #pose
            new_data_info['cam2img'] = scene['cam2img'] #intrinsic
            new_data_info['axis_align_matrix'] = scene['axis_align_matrix']
            new_data_info['depth_scale'] = 1000
            data.update(new_data_info)
        return data_list
    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        This method should return dict or list of dict. Each dict or list
        contains the data information of a training sample. If the protocol of
        the sample annotations is changed, this function can be overridden to
        update the parsing logic while keeping compatibility.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            list or list[dict]: Parsed annotation.
        """
        for prefix_key, prefix in self.data_prefix.items():
            assert prefix_key in raw_data_info, (
                f'raw_data_info: {raw_data_info} dose not contain prefix key'
                f'{prefix_key}, please check your data_prefix.')
            if isinstance(raw_data_info[prefix_key], list):
                raw_data_info[prefix_key] = [
                    join_path(prefix, item) for item in raw_data_info[prefix_key]
                ]
            else:
                raw_data_info[prefix_key] = join_path(prefix,
                                                    raw_data_info[prefix_key])
        return raw_data_info
    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        If the annotation file does not follow `OpenMMLab 2.0 format dataset
        <https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html>`_ .
        The subclass must override this method for load annotations. The meta
        information of annotation file will be overwritten :attr:`METAINFO`
        and ``metainfo`` argument of constructor.

        Returns:
            list[dict]: A list of annotation.
        """  # noqa: E501
        # `self.ann_file` denotes the absolute annotation file path if
        # `self.root=None` or relative path if `self.root=/path/to/data/`.
        raw_data_list = load(self.ann_file)
        raw_data_list = self.merge_scannet_info(raw_data_list)
        # load and parse data_infos.
        data_list = []
        for raw_data_info in raw_data_list:
            # parse raw data information to target format
            data_info = self.parse_data_info(raw_data_info)
            if isinstance(data_info, dict):
                # For image tasks, `data_info` should information if single
                # image, such as dict(img_path='xxx', width=360, ...)
                data_list.append(data_info)
            elif isinstance(data_info, list):
                # For video tasks, `data_info` could contain image
                # information of multiple frames, such as
                # [dict(video_path='xxx', timestamps=...),
                #  dict(video_path='xxx', timestamps=...)]
                for item in data_info:
                    if not isinstance(item, dict):
                        raise TypeError('data_info must be list of dict, but '
                                        f'got {type(item)}')
                data_list.extend(data_info)
            else:
                raise TypeError('data_info should be a dict or list of dict, '
                                f'but got {type(data_info)}')

        return data_list
if __name__ == '__main__':
    from mmengine.registry import init_default_scope
    init_default_scope('embodiedbot')  # 确保 scope 初始化
    pipeline = [
        dict(type='MultiViewPipeline',
             n_images=20,
             sampling_strategy='uniform',
             transforms=[
                 dict(type='LoadImageFromFile'),
                 dict(type='LoadDepthFromFile'),
             ]),
        dict(type='ConvertDepthToCoordinate'),
        dict(type='PackBotData', keys=['img', 'depth', 'global_coords', 'metadata', 'id', 'input_id', 'target', 'box_input', 'box_output']),
    ]
    dataset = ScanNetTaskBaseDataset(data_root = './data',
                                    ann_file='./processed/scanqa_train_llava_style.json',
                                    scannet_info_file='./scannet/mltiview_infos/mv_scannetv2_infos_train.pkl',
                                    pipeline=pipeline,
                                    )
    dataset.full_init()
    print(dataset[0]['img'].shape)