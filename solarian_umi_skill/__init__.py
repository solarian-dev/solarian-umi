"""A UMI diffusion-policy skill.

``UMISkill`` reads its policy, wiring, and motion parameters from the yaml
passed as ``params["config"]``. ``solarian install`` copies
``templates/skills/umi.yaml`` to ``conf/skills/umi.yaml`` the first time; copy that file
to define another policy.

A policy trained with targets (a box, point or mask on the wrist image) reads
them from camera fields named by the yaml's ``targets``
(``templates/skills/umi_target.yaml``).
"""

from solarian_umi_skill.skill import UMISkill

__all__ = ["UMISkill"]
