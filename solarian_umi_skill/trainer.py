"""Local UMI training.

``solarian train`` resolves ``policy: umi`` to :class:`UmiTrainer`. The
training code is the UMI workspace copied into this package. Hydra
composition happens in :mod:`solarian_umi_skill.training`, because Solarian
has already initialized Hydra for the project conf.
"""

from __future__ import annotations

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
    batch_size: int = 64
    num_epochs: int = 50
    num_workers: int = 8
    optimizer_lr: float = 3.0e-4
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
    return [
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
        *cfg.overrides,
    ]


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
        from solarian_umi_skill.training import run_training

        run_training(cfg, output_dir)
