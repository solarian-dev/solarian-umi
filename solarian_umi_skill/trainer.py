"""Local UMI training.

``solarian train`` resolves ``policy: umi`` to :class:`UmiTrainer`. The
training code is the UMI workspace copied into this package. Hydra
composition happens in :mod:`solarian_umi_skill.training`, because Solarian
has already initialized Hydra for the project conf.
"""

from __future__ import annotations

import math
import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from solarian.core.ai.system1.policy_trainer import ModalSpec, PolicyTrainer


class UmiLoggingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    mode: str = "online"
    project: str = "umi"


class UmiDatasetEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    weight: float = Field(default=1.0, gt=0)


class UmiTrainConfig(BaseModel):
    """Fields of ``conf/train-umi.yaml``. Extra keys are kept and ignored."""

    model_config = ConfigDict(extra="allow")

    policy: str
    job_name: str
    output_dir: str | None = None
    config_name: str = "train_diffusion_unet_timm_umi_cup_wild"
    dataset_path: str | None = None
    # Instead of dataset_path: train on several .zarr.zip files, each sample
    # drawn from entry i with probability weight_i / sum(weights) whatever
    # its size. Frames are sampled, not copied. Validation comes from the
    # first entry; the normalizer is fit on all of them.
    datasets: list[UmiDatasetEntry] | None = None
    # Validate on this .zarr.zip and train on all of dataset_path. Unset: a
    # random task.dataset.val_ratio share of dataset_path's episodes.
    val_dataset_path: str | None = None
    batch_size: int = 64
    num_epochs: int = 50
    # Instead of num_epochs: train for this many optimizer steps, in
    # "epochs" of steps_per_epoch randomly drawn batches. An epoch is then
    # only the cadence of validation, sampling and checkpoints.
    num_train_steps: int | None = None
    steps_per_epoch: int | None = None
    num_workers: int = 8
    optimizer_lr: float = 3.0e-4
    # Fine-tuning: start from this checkpoint's weights with a fresh optimizer,
    # LR schedule and epoch count. The normalizer is recomputed from the
    # training data. The architecture must match, except that a cls checkpoint
    # can start a box_pool run.
    init_from_ckpt: str | None = None
    # Continue an interrupted run from <output_dir>/checkpoints/latest.ckpt at
    # its next epoch; num_epochs stays the run's total. Takes precedence over
    # init_from_ckpt.
    resume: bool = False
    # How the image features condition the policy. Only for configs with a
    # mask_channel bbox. cls: the ViT summary token. box_pool: adds the patch
    # tokens under the box and its corners. A cls init_from_ckpt is widened
    # with zero weights for those inputs. null keeps the config's choice.
    bbox_conditioning: Literal["cls", "box_pool"] | None = None
    logging: UmiLoggingConfig = Field(default_factory=UmiLoggingConfig)
    overrides: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_data_and_budget_source(self) -> "UmiTrainConfig":
        if self.dataset_path and self.datasets:
            raise ValueError("Set dataset_path or datasets, not both.")
        if self.datasets == []:
            raise ValueError("datasets needs at least one entry.")
        if (self.num_train_steps is None) != (self.steps_per_epoch is None):
            raise ValueError("num_train_steps and steps_per_epoch go together.")
        if self.num_train_steps is not None and "num_epochs" in self.model_fields_set:
            raise ValueError("Set num_epochs or num_train_steps, not both.")
        return self

    @property
    def dataset_paths(self) -> list[str]:
        if self.datasets:
            return [entry.path for entry in self.datasets]
        return [self.dataset_path] if self.dataset_path else []


def _hydra_value(value: object) -> str:
    """A scalar in Hydra override syntax.

    Numbers stay unquoted so Hydra keeps their type. Strings are quoted:
    a dataset path or output dir may contain spaces or ``=``.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", r"\'") + "'"


def _hydra_override(key: str, value: object) -> str:
    return f"{key}={_hydra_value(value)}"


def train_overrides(cfg: UmiTrainConfig, output_dir: str) -> list[str]:
    """Hydra overrides applied on top of the shipped UMI config.

    ``output_dir`` is ``models/<job_name>``. The workspace writes
    ``checkpoints/latest.ckpt`` under it, which is the path the matching
    skill yaml's ``policy_server.checkpoint`` names.
    """
    paths = [os.path.expanduser(p) for p in cfg.dataset_paths]
    if cfg.num_train_steps is None:
        num_epochs = cfg.num_epochs
    else:
        num_epochs = math.ceil(cfg.num_train_steps / cfg.steps_per_epoch)
    overrides = [
        _hydra_override("task.dataset_path", paths[0]),
        _hydra_override("batch_size", cfg.batch_size),
        _hydra_override("training.num_epochs", num_epochs),
        _hydra_override("dataloader.num_workers", cfg.num_workers),
        _hydra_override("val_dataloader.num_workers", cfg.num_workers),
        _hydra_override("optimizer.lr", cfg.optimizer_lr),
        _hydra_override("logging.mode", cfg.logging.mode),
        _hydra_override("logging.project", cfg.logging.project),
        _hydra_override("hydra.run.dir", output_dir),
        _hydra_override("num_gpus", 1),
    ]
    if cfg.datasets and len(cfg.datasets) > 1:
        entries = ",".join(
            f"{{path:{_hydra_value(path)},weight:{_hydra_value(entry.weight)}}}"
            for path, entry in zip(paths, cfg.datasets))
        overrides.append(f"+task.dataset_mix=[{entries}]")
    if cfg.steps_per_epoch is not None:
        overrides.append(_hydra_override(
            "+task.samples_per_epoch", cfg.steps_per_epoch * cfg.batch_size))
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

    @classmethod
    def modal_spec(cls) -> ModalSpec:
        # Pinned to the environment the deployed checkpoints were trained and
        # evaluated in.
        return ModalSpec(
            python_version="3.10",
            apt_packages=["libgl1", "libglib2.0-0", "ffmpeg"],
            pip_packages=[
                "torch==2.14.1",
                "torchvision==0.29.1",
                "timm==1.0.30",
                "diffusers==0.40.0",
                "accelerate==1.15.0",
                "wandb==0.30.0",
                "hydra-core==1.3.2",
                "omegaconf==2.3.1",
                "einops==0.8.2",
                "dill==0.4.1",
                "zarr==2.18.3",
                "numcodecs==0.13.1",
                "imagecodecs==2025.3.30",
                "pandas==2.3.3",
                "numba==0.68.0",
                "numpy==2.2.6",
                "scipy==1.15.3",
                "av==17.1.0",
                "tqdm",
            ],
            local_python_packages=["solarian", "solarian_umi_skill"],
        )

    def train(self) -> None:
        cfg: UmiTrainConfig = self.cfg
        if not cfg.dataset_paths:
            raise ValueError(
                "Training config must set dataset_path to a .zarr.zip dataset, "
                "or datasets to several."
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
        for path in cfg.dataset_paths:
            if not os.path.isfile(os.path.expanduser(path)):
                raise FileNotFoundError(f"dataset not found: {path}")
        # Without latest.ckpt the workspace would neither resume nor apply
        # init_from_ckpt, and silently start from the pretrained backbone.
        latest = os.path.join(output_dir, "checkpoints", "latest.ckpt")
        if cfg.resume and not os.path.isfile(latest):
            raise FileNotFoundError(f"resume: no checkpoint at {latest}")
        from solarian_umi_skill.training import run_training

        run_training(cfg, output_dir)
