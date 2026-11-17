"""Run the UMI workspace on a training file's UMI config.

Kept apart from :mod:`solarian_umi.trainer` so the torch and workspace
imports only happen when training starts.
"""

from __future__ import annotations

from typing import Any, Dict


def run_training(umi_config: Dict[str, Any], output_dir: str) -> None:
    import hydra
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(umi_config)
    cls = hydra.utils.get_class(cfg._target_)
    cls(cfg, output_dir=output_dir).run()
