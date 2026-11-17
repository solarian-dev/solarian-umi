"""Local UMI training.

``solarian train`` resolves ``trainer: umi`` to :class:`UmiTrainer`. A
training file (``conf/training/<name>.yaml``) is the UMI training config
written as ``python train.py`` reads it in the UMI repo, with the task inlined
under ``task:``, plus the Solarian keys ``trainer``, ``job_name``,
``output_dir`` and ``modal``. The training code is the UMI workspace copied
into this package.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from solarian.core.ai.system1.policy_trainer import ModalSpec, PolicyTrainer
from solarian_umi.targets import target_keys

SOLARIAN_KEYS = {"trainer", "job_name", "output_dir", "modal"}


def _expand(path: Optional[str]) -> Optional[str]:
    # The UMI workspace opens dataset and checkpoint paths as given.
    return os.path.expanduser(path) if path else path


class UmiDatasetEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    # Each sample is drawn from entry i with probability weight_i / sum(weights).
    weight: float = Field(gt=0)

    _expand_path = field_validator("path")(_expand)


class UmiTaskDataset(BaseModel):
    model_config = ConfigDict(extra="allow")

    val_dataset_path: Optional[str] = Field(
        description="Validated on in full, and every training episode trains. "
                    "None: validates on a random val_ratio share of the first dataset's episodes.")

    _expand_path = field_validator("val_dataset_path")(_expand)


class UmiTask(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset_path: str
    dataset_mix: List[UmiDatasetEntry] = Field(min_length=1)
    dataset: UmiTaskDataset
    shape_meta: Dict[str, Any]

    _expand_path = field_validator("dataset_path")(_expand)


class UmiTraining(BaseModel):
    model_config = ConfigDict(extra="allow")

    # Continue from <output_dir>/checkpoints/latest.ckpt.
    resume: bool
    init_from_ckpt: Optional[str] = Field(
        description="Weights to start from, with a fresh optimizer, LR schedule and epoch count. "
                    "None: only the visual backbone starts from weights (policy.obs_encoder.pretrained).")

    _expand_path = field_validator("init_from_ckpt")(_expand)


class UmiTrainConfig(BaseModel):
    """A UMI training file, resolved.

    Every top-level key of the UMI config is declared, so a misspelled or stray
    key fails validation instead of being ignored. Below the top level only the
    keys Solarian reads are typed; the rest belongs to the UMI workspace.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    trainer: str
    job_name: str
    # None: solarian train uses models/<job_name>.
    output_dir: Optional[str]
    # Read by solarian modal-train; ignored by solarian train.
    modal: Optional[Dict[str, Any]]

    name: str
    target: str = Field(alias="_target_")
    task: UmiTask
    shape_meta: Dict[str, Any]
    exp_name: str
    batch_size: int
    visual_encoder: str
    visual_encoder_tag: str
    model_type: str
    num_gpus: int
    mirror: Any
    real_data_size: Any
    sim_data_size: Any
    bbox: Any
    task_name: Any
    bbox_mode: Any
    n_action_steps: int
    policy: Dict[str, Any]
    ema: Dict[str, Any]
    dataloader: Dict[str, Any]
    val_dataloader: Dict[str, Any]
    optimizer: Dict[str, Any]
    training: UmiTraining
    logging: Dict[str, Any]
    checkpoint: Dict[str, Any]
    multi_run: Dict[str, Any]

    def umi_config(self) -> Dict[str, Any]:
        """The UMI part of the file, as the workspace reads it."""
        return self.model_dump(by_alias=True, exclude=SOLARIAN_KEYS)


def is_dataset(dataset_path: str) -> bool:
    """A ``.zarr.zip``, a ``.solarian.zip`` or a session directory exists here."""
    from solarian_umi.diffusion_policy.dataset.solarian_session import is_session

    return os.path.isfile(os.path.expanduser(dataset_path)) or is_session(dataset_path)


def has_target_array(dataset_path: str, key: str) -> bool:
    """Whether a dataset has target ``key`` for every frame, without loading it."""
    from solarian_umi.diffusion_policy.dataset.solarian_session import (
        has_target, is_session)

    if is_session(dataset_path):
        return has_target(dataset_path, key)
    import zipfile

    with zipfile.ZipFile(os.path.expanduser(dataset_path)) as zf:
        return f"data/{key}/.zarray" in zf.namelist()


def _missing_target_hint(key: str, kind: str, dataset_path: str) -> str:
    from solarian_umi.diffusion_policy.dataset.solarian_session import (
        episodes_missing_target, is_session)

    if is_session(dataset_path):
        op = "track_point" if kind == "point" else "track"
        missing = [Path(episode).name for episode in episodes_missing_target(dataset_path, key)]
        shown = ", ".join(missing[:10]) + (f" and {len(missing) - 10} more" if len(missing) > 10 else "")
        return (f"every episode's targets/camera0.rgb.npz needs a {key!r} array, and episodes {shown} "
                f"have none; annotate them with the {op} op (or solarian_session.write_targets), "
                f"or delete them")
    if kind == "mask":
        return "it needs a data/mask array; add it with solarian_base scripts/zarr_add_masks.py"
    return f"it needs a data/{key} array"


class UmiTrainer(PolicyTrainer):
    title = "UMI Policy Trainer"

    @classmethod
    def config_class(cls) -> type:
        return UmiTrainConfig

    @classmethod
    def resolvers(cls) -> Dict[str, Callable[..., Any]]:
        from solarian_umi.diffusion_policy.dataset_name import parse_dataset_field

        return {"eval": eval, "parse_dataset_field": parse_dataset_field}

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
            local_python_packages=["solarian", "solarian_umi"],
        )

    def check_inputs(self, skip: Callable[[str], bool] = lambda path: False) -> None:
        cfg: UmiTrainConfig = self.cfg
        task, init_from_ckpt = cfg.task, cfg.training.init_from_ckpt
        val_dataset = task.dataset.val_dataset_path
        datasets = [path for path in (entry.path for entry in task.dataset_mix) if not skip(path)]
        if val_dataset and skip(val_dataset):
            val_dataset = None
        # The workspace only warns on a missing init checkpoint and then trains
        # from the pretrained backbone.
        if init_from_ckpt and not skip(init_from_ckpt) and not os.path.isfile(os.path.expanduser(init_from_ckpt)):
            raise FileNotFoundError(f"training.init_from_ckpt not found: {init_from_ckpt}")
        if val_dataset and not is_dataset(val_dataset):
            raise FileNotFoundError(f"task.dataset.val_dataset_path not found: {val_dataset}")
        for path in datasets:
            if not is_dataset(path):
                raise FileNotFoundError(f"task.dataset_mix: dataset not found: {path}")
        for key, kind in target_keys(task.shape_meta["obs"]).items():
            for path in datasets + [p for p in [val_dataset] if p]:
                if not has_target_array(path, key):
                    raise ValueError(f"{cfg.job_name} observes the {kind} target {key!r}, which "
                                     f"{path} lacks: {_missing_target_hint(key, kind, path)}")

    def train(self) -> None:
        from solarian_umi import training

        cfg: UmiTrainConfig = self.cfg
        output_dir = self.output_dir or cfg.output_dir
        if not output_dir:
            raise ValueError(
                "output_dir is not set. solarian train sets it to models/<job_name>."
            )
        # Without latest.ckpt the workspace would neither resume nor apply
        # init_from_ckpt, and silently start from the pretrained backbone.
        latest = os.path.join(output_dir, "checkpoints", "latest.ckpt")
        if cfg.training.resume and not os.path.isfile(latest):
            raise FileNotFoundError(f"training.resume: no checkpoint at {latest}")
        training.run_training(cfg.umi_config(), output_dir)
