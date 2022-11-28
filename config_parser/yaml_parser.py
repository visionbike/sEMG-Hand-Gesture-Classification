from typing import Union
import shutil
from abc import ABC
from pathlib import Path
import torch
from yacs.config import CfgNode as cn
from config_parser.base_parser import *
from data.preprocessing.utils import *

__all__ = ['YamlConfigParser']

VALID_TYPES = {tuple, list, str, int, float, bool, None, torch.Tensor}


def convert_to_dict(cfg_node: cn, keys: list) -> Union[cn, dict]:
    """
    Convert a config node to dictionary.

    :param cfg_node: the input configuration node CfgNode.
    :param keys: the key list.
    """

    if not isinstance(cfg_node, cn):
        if type(cfg_node) not in VALID_TYPES:
            print(f"Key {keys} with value {cfg_node} is not a valid type; valid types: {VALID_TYPES}")
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, keys + [k])
        return cfg_dict


class YamlConfigParser(BaseConfigParser, ABC):
    """
    The custom configuration parser for YAML config files.
    """

    def __init__(self, description: str):
        """

        @param description: the description for parser.
        """

        super(YamlConfigParser, self).__init__()
        self.parser.description = description
        # initial parser
        self.init_args()

    def _add_arguments(self) -> None:
        self.parser.add_argument('--cfg',
                                 default='./cfgs/config_baseline_nina5_ver1.yaml', help='the path of yaml config file.')
        self.parser.add_argument('--log',
                                 action='store_true', default=False, help='Applying online logging')

    def _print_args(self) -> None:
        super(YamlConfigParser, self)._print_args()

    def init_args(self) -> None:
        """
        Initial config arguments.
        """

        self._add_arguments()
        self.args = self.parser.parse_args()

    def parse(self) -> cn:
        """
        Load configs from file.
        """

        print('### Load the YAML config file...')
        with open(self.args.cfg, 'r') as f:
            cfgs = cn.load_cfg(f)
            print(f'Successfully loading the config YAML file!')

        if ('DataConfig' in cfgs.keys()) and ('DataProcessConfig' in cfgs.keys()):
            # setup data path
            cfgs.DataConfig.path = f'./datasets/{cfgs.DataConfig.name}/processed'
            cfgs.DataConfig.path += f'/ver{cfgs.DataProcessConfig.ver}_'
            if ('use_imu' in cfgs.DataProcessConfig.keys()) and cfgs.DataProcessConfig.use_imu:
                cfgs.DataConfig.path += 'imu_'
            if cfgs.DataProcessConfig.use_rectify:
                cfgs.DataConfig.path += 'rectify_'
            if cfgs.DataProcessConfig.use_butter:
                cfgs.DataConfig.path += 'butter_'
            cfgs.DataConfig.path += f's{cfgs.DataProcessConfig.step_size}_'
            cfgs.DataConfig.path += f'w{cfgs.DataProcessConfig.window_size}_'
            if cfgs.DataProcessConfig.first:
                cfgs.DataConfig.path += 'first_'
            else:
                cfgs.DataConfig.path += 'major_'
            if cfgs.DataProcessConfig.use_rest_label:
                cfgs.DataConfig.path += 'rest'
            else:
                cfgs.DataConfig.path += 'no_rest'
            cfgs.DataConfig.use_rest_label = cfgs.DataProcessConfig.use_rest_label
            if not cfgs.DataConfig.use_rest_label:
                cfgs.DataConfi.num_classes -= 1

        if 'NetworkConfig' in cfgs.keys():
            if cfgs.DataConfig.msize is not None:
                cfgs.NetworkConfig.in_dims = cfgs.DataProcessConfig.window_size - cfgs.DataConfig.msize + 1
            else:
                cfgs.NetworkConfig.in_dims = cfgs.DataProcessConfig.window_size
            cfgs.NetworkConfig.num_classes = cfgs.DataConfig.num_classes

        if 'ExpConfig' in cfgs.keys():
            # name
            cfgs.ExpConfig.name = (cfgs.DataConfig.name + '_' +
                                   'ver' + str(cfgs.DataProcessConfig.ver) + '_' +
                                   cfgs.NetworkConfig.name + '_' +
                                   cfgs.NetworkConfig.attention.name + '_' +
                                   'exp' + str(cfgs.ExpConfig.experiment))
            # create `experiments` directory
            exp_dir = Path('./experiments') / cfgs.ExpConfig.name
            exp_dir.mkdir(parents=True, exist_ok=True)
            # copy the config file to current experiment directory
            shutil.copyfile(self.args.cfg, exp_dir / f'config_{cfgs.ExpConfig.phase}.yaml')
            # setup experiment configs
            cfgs.ExpConfig.exp_dir = str(exp_dir)
            # setup logging
            cfgs.ExpConfig.logging = self.args.log

            # setup for train/test process
            if cfgs.ExpConfig.phase == 'train':
                # setup loss configs
                cfgs.LossConfig.num_classes = cfgs.DataConfig.num_classes
                cfgs.LossConfig.weights = get_class_weights(
                    cfgs.LossConfig.num_classes,
                    cfgs.LossConfig.weights)

        # setup metric config
        cfgs.MetricConfig = cn()
        cfgs.MetricConfig.num_classes = cfgs.DataConfig.num_classes

        # print configurations
        print(f"### Configurations:\n{cfgs}")
        return cfgs
