"""Compose the shipped UMI Hydra config and run its workspace.

Solarian has already initialized Hydra on the project conf by the time
``train()`` runs. That config directory does not contain the UMI task
group or the ``eval`` and ``parse_dataset_field`` resolvers, so Hydra is
cleared and initialized again on the config copied into this package.
``train()`` is the rest of the process.
"""

from __future__ import annotations

from pathlib import Path

from solarian_umi_skill.trainer import UmiTrainConfig, train_overrides


def config_dir() -> Path:
    return Path(__file__).resolve().parent / "diffusion_policy" / "config"


def run_training(cfg: UmiTrainConfig, output_dir: str) -> None:
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    from solarian_umi_skill.diffusion_policy.dataset_name import parse_dataset_field

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver(
        "parse_dataset_field", parse_dataset_field, replace=True
    )
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base=None, config_dir=str(config_dir())):
        composed = compose(
            config_name=cfg.config_name,
            overrides=train_overrides(cfg, output_dir),
        )
    OmegaConf.resolve(composed)
    cls = hydra.utils.get_class(composed._target_)
    cls(composed, output_dir=output_dir).run()
