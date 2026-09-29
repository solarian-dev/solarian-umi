"""A UMI diffusion-policy skill.

``UMISkill`` reads its policy, wiring, and motion parameters from the yaml
passed as ``params["config"]``. ``solarian install`` copies
``templates/umi.yaml`` to ``conf/umi.yaml`` the first time; copy that file
to define another policy.
"""

from solarian_umi_skill.skill import UMISkill

__all__ = ["UMISkill"]
