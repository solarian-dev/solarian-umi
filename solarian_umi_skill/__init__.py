"""A UMI diffusion-policy skill.

``UMISkill`` reads its policy, wiring, and motion parameters from the yaml
passed as ``params["config"]``. ``solarian install`` copies
``templates/umi.yaml`` to ``conf/umi.yaml`` the first time; copy that file
to define another policy.

``UMIPnPSkill`` is the point-and-pick variant: the policy also takes the
camera's tracked target box (``templates/umi_pnp.yaml``).
"""

from solarian_umi_skill.pnp_skill import UMIPnPSkill
from solarian_umi_skill.skill import UMISkill

__all__ = ["UMIPnPSkill", "UMISkill"]
