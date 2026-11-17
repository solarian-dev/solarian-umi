# Adapted from Universal Manipulation Interface.
# Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab.
# https://github.com/real-stanford/universal_manipulation_interface
# Licensed under the MIT License. The full text is LICENSE.umi
# beside this package.

from typing import Dict
import torch
import torch.nn as nn
from solarian_umi.diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from solarian_umi.diffusion_policy.model.common.normalizer import LinearNormalizer

class BaseImagePolicy(ModuleAttrMixin):
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], fixed_action_prefix: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        fixed_action_prefix:
            B, Tp, Da
        return: B,Ta,Da
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()
