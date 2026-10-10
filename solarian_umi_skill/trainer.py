"""Local UMI training.

``solarian train`` resolves ``policy: umi`` to :class:`UmiTrainer`. The
training code is the UMI workspace copied into this package. Hydra
composition happens in :mod:`solarian_umi_skill.training`, because Solarian
has already initialized Hydra for the project conf.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from solarian.core.ai.system1.policy_trainer import PolicyTrainer


class UmiLoggingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    mode: str = "online"
    project: str = "umi"


class UmiTrainConfig(BaseModel):
    """Fields of ``conf/train-umi.yaml``. Extra keys are kept and ignored."""

    model_config = ConfigDict(extra="allow")

    policy: str
    job_name: str
    output_dir: str | None = None
    config_name: str = "train_diffusion_unet_timm_umi_cup_wild"
    dataset_path: str | None = None
    # Validate on this .zarr.zip and train on all of dataset_path. Unset: a
    # random task.dataset.val_ratio share of dataset_path's episodes.
    val_dataset_path: str | None = None
    batch_size: int = 64
    num_epochs: int = 50
    num_workers: int = 8
    optimizer_lr: float = 3.0e-4
    # Fine-tuning: start from this checkpoint's weights with a fresh optimizer,
    # LR schedule and epoch count. The normalizer is recomputed from
    # dataset_path. The architecture (including bbox_conditioning) must match.
    init_from_ckpt: str | None = None
    # Continue an interrupted run from <output_dir>/checkpoints/latest.ckpt at
    # its next epoch; num_epochs stays the run's total. Takes precedence over
    # init_from_ckpt.
    resume: bool = False
    # How the image features condition the policy. Only for configs with a
    # mask_channel bbox. cls: the ViT summary token. box_pool: adds the patch
    # tokens under the box and its corners; it has a different width, so a
    # cls checkpoint cannot initialize it. null keeps the config's choice.
    bbox_conditioning: Literal["cls", "box_pool"] | None = None
    logging: UmiLoggingConfig = Field(default_factory=UmiLoggingConfig)
    overrides: list[str] = Field(default_factory=list)


def _hydra_override(key: str, value: object) -> str:
    """One Hydra ``key=value`` token.

    Numbers stay unquoted so Hydra keeps their type. Strings are quoted:
    a dataset path or output dir may contain spaces or ``=``.
    """
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, (int, float)):
        text = str(value)
    else:
        text = "'" + str(value).replace("'", r"\'") + "'"
    return f"{key}={text}"


def train_overrides(cfg: UmiTrainConfig, output_dir: str) -> list[str]:
    """Hydra overrides applied on top of the shipped UMI config.

    ``output_dir`` is ``models/<job_name>``. The workspace writes
    ``checkpoints/latest.ckpt`` under it, which is the path the matching
    skill yaml's ``policy_server.checkpoint`` names.
    """
    overrides = [
        _hydra_override("task.dataset_path", cfg.dataset_path),
        _hydra_override("batch_size", cfg.batch_size),
        _hydra_override("training.num_epochs", cfg.num_epochs),
        _hydra_override("dataloader.num_workers", cfg.num_workers),
        _hydra_override("val_dataloader.num_workers", cfg.num_workers),
        _hydra_override("optimizer.lr", cfg.optimizer_lr),
        _hydra_override("logging.mode", cfg.logging.mode),
        _hydra_override("logging.project", cfg.logging.project),
        _hydra_override("hydra.run.dir", output_dir),
        _hydra_override("num_gpus", 1),
    ]
    if cfg.val_dataset_path:
        overrides.append(_hydra_override(
            "task.dataset.val_dataset_path", os.path.expanduser(cfg.val_dataset_path)))
    if cfg.resume:
        overrides.append(_hydra_override("training.resume", True))
    if cfg.init_from_ckpt:
        overrides.append(_hydra_override(
            "+training.init_from_ckpt", os.path.expanduser(cfg.init_from_ckpt)))
    if cfg.bbox_conditioning == "cls":
        overrides.append("policy.obs_encoder.feature_aggregation=null")
    elif cfg.bbox_conditioning == "box_pool":
        overrides.append("policy.obs_encoder.feature_aggregation=box_pool")
    return overrides + cfg.overrides


class UmiTrainer(PolicyTrainer):
    title = "UMI Policy Trainer"

    @classmethod
    def config_class(cls) -> type:
        return UmiTrainConfig

    def train(self) -> None:
        cfg: UmiTrainConfig = self.cfg
        if not cfg.dataset_path:
            raise ValueError(
                "Training config must set dataset_path to a .zarr.zip dataset."
            )
        output_dir = self.output_dir or cfg.output_dir
        if not output_dir:
            raise ValueError(
                "output_dir is not set. solarian train sets it to models/<job_name>."
            )
        # Checked before the training set loads. The workspace only warns on a
        # missing init checkpoint and then trains from the pretrained backbone.
        for field in ("init_from_ckpt", "val_dataset_path"):
            path = getattr(cfg, field)
            if path and not os.path.isfile(os.path.expanduser(path)):
                raise FileNotFoundError(f"{field} not found: {path}")
        # Without latest.ckpt the workspace would neither resume nor apply
        # init_from_ckpt, and silently start from the pretrained backbone.
        latest = os.path.join(output_dir, "checkpoints", "latest.ckpt")
        if cfg.resume and not os.path.isfile(latest):
            raise FileNotFoundError(f"resume: no checkpoint at {latest}")
        from solarian_umi_skill.training import run_training

        run_training(cfg, output_dir)
