# Adapted from Universal Manipulation Interface.
# Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab.
# https://github.com/real-stanford/universal_manipulation_interface
# Licensed under the MIT License. The full text is LICENSE.umi
# beside this package.

from solarian_umi.diffusion_policy.policy.base_image_policy import BaseImagePolicy
from solarian_umi.diffusion_policy.env_runner.base_image_runner import BaseImageRunner

class RealPushTImageRunner(BaseImageRunner):
    def __init__(self,
            output_dir):
        super().__init__(output_dir)
    
    def run(self, policy: BaseImagePolicy):
        return dict()
