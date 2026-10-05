"""UmiTrainer builds Hydra overrides. It does not run training."""

from pathlib import Path

import pytest
import yaml

from solarian_umi_skill.trainer import UmiTrainConfig, UmiTrainer, train_overrides

TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "solarian_umi_skill"
    / "templates"
    / "train-umi.yaml"
)


def _cfg(**overrides) -> UmiTrainConfig:
    data = dict(policy="umi", job_name="cup_arrange", dataset_path="datasets/cup.zarr.zip")
    data.update(overrides)
    return UmiTrainConfig.model_validate(data)


def test_template_selects_the_umi_trainer():
    loaded = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    cfg = UmiTrainConfig.model_validate(loaded)

    assert loaded["policy"] == "umi"
    assert "umi_root" not in loaded
    assert "umi_root" not in UmiTrainConfig.model_fields
    assert cfg.policy == "umi"
    assert cfg.config_name == "train_diffusion_unet_timm_umi_cup_wild"


def test_two_tasks_get_different_datasets_and_output_dirs(tmp_path):
    cup = train_overrides(
        _cfg(job_name="cup_arrange", dataset_path="datasets/cup.zarr.zip"),
        str(tmp_path / "models" / "cup_arrange"),
    )
    pick = train_overrides(
        _cfg(job_name="pick_object", dataset_path="datasets/pick.zarr.zip"),
        str(tmp_path / "models" / "pick_object"),
    )

    assert "task.dataset_path='datasets/cup.zarr.zip'" in cup
    assert "task.dataset_path='datasets/pick.zarr.zip'" in pick
    assert f"hydra.run.dir='{tmp_path / 'models' / 'cup_arrange'}'" in cup
    assert f"hydra.run.dir='{tmp_path / 'models' / 'pick_object'}'" in pick
    assert "num_gpus=1" in cup
    assert "batch_size=64" in cup
    assert "optimizer.lr=0.0003" in cup


def test_overrides_are_appended_and_paths_with_equals_are_quoted(tmp_path):
    cmd = train_overrides(
        _cfg(
            dataset_path="runs/epoch=0170.zarr.zip",
            overrides=["n_action_steps=8", "+training.init_from_ckpt='/ckpt/epoch=1.ckpt'"],
        ),
        str(tmp_path / "out"),
    )

    assert "task.dataset_path='runs/epoch=0170.zarr.zip'" in cmd
    assert cmd[-2:] == [
        "n_action_steps=8",
        "+training.init_from_ckpt='/ckpt/epoch=1.ckpt'",
    ]


def test_missing_dataset_path_does_not_start_training(tmp_path, monkeypatch):
    def fail(*_a, **_k):
        raise AssertionError("training should not start")

    monkeypatch.setattr("solarian_umi_skill.training.run_training", fail)
    with pytest.raises(ValueError, match="dataset_path"):
        UmiTrainer(_cfg(dataset_path=None), output_dir=str(tmp_path)).train()
