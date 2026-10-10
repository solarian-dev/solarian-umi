"""UmiTrainer builds Hydra overrides. It does not run training."""

import os
from pathlib import Path

import pytest
import yaml

from solarian_umi_skill.trainer import UmiTrainConfig, UmiTrainer, train_overrides
from solarian_umi_skill.training import config_dir

TEMPLATES = Path(__file__).resolve().parents[1] / "solarian_umi_skill" / "templates"
TEMPLATE = TEMPLATES / "train-umi.yaml"
PICK_TEMPLATE = TEMPLATES / "train-umi-pick.yaml"


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


def test_pick_template_names_a_shipped_config():
    loaded = yaml.safe_load(PICK_TEMPLATE.read_text(encoding="utf-8"))
    cfg = UmiTrainConfig.model_validate(loaded)

    assert (config_dir() / f"{cfg.config_name}.yaml").is_file()
    assert cfg.bbox_conditioning == "cls"
    assert cfg.init_from_ckpt is None


def test_fine_tune_and_conditioning_become_hydra_overrides(tmp_path):
    cmd = train_overrides(
        _cfg(init_from_ckpt="~/ckpts/epoch=0170.ckpt", bbox_conditioning="box_pool",
             val_dataset_path="~/data/side_val.zarr.zip", overrides=["n_action_steps=8"]),
        str(tmp_path / "out"),
    )

    home = os.path.expanduser("~")
    assert f"+training.init_from_ckpt='{home}/ckpts/epoch=0170.ckpt'" in cmd
    assert f"task.dataset.val_dataset_path='{home}/data/side_val.zarr.zip'" in cmd
    assert "policy.obs_encoder.feature_aggregation=box_pool" in cmd
    assert cmd[-1] == "n_action_steps=8"


def test_unset_conditioning_keeps_the_config_choice(tmp_path):
    cmd = train_overrides(_cfg(), str(tmp_path / "out"))

    assert not any("feature_aggregation" in o or "init_from_ckpt" in o for o in cmd)
    assert "policy.obs_encoder.feature_aggregation=null" in train_overrides(
        _cfg(bbox_conditioning="cls"), str(tmp_path / "out"))


@pytest.mark.parametrize("field", ["init_from_ckpt", "val_dataset_path"])
def test_missing_input_file_does_not_start_training(tmp_path, monkeypatch, field):
    def fail(*_a, **_k):
        raise AssertionError("training should not start")

    monkeypatch.setattr("solarian_umi_skill.training.run_training", fail)
    with pytest.raises(FileNotFoundError, match=field):
        UmiTrainer(_cfg(**{field: str(tmp_path / "missing")}), output_dir=str(tmp_path)).train()


def test_resume_needs_the_run_latest_checkpoint(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr("solarian_umi_skill.training.run_training",
                        lambda cfg, out: started.append(train_overrides(cfg, out)))
    trainer = UmiTrainer(_cfg(resume=True), output_dir=str(tmp_path))

    with pytest.raises(FileNotFoundError, match="latest.ckpt"):
        trainer.train()

    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "latest.ckpt").write_bytes(b"")
    trainer.train()
    assert "training.resume=true" in started[0]


def test_pick_config_composes_with_the_box_and_the_selected_conditioning(tmp_path):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    from solarian_umi_skill.diffusion_policy.dataset_name import parse_dataset_field

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("parse_dataset_field", parse_dataset_field, replace=True)
    GlobalHydra.instance().clear()
    cfg = _cfg(config_name="train_diffusion_unet_timm_umi_pick_bbox",
               dataset_path="pick_lego_pile-no_mirror-real_368-bbox.zarr.zip",
               bbox_conditioning="box_pool", val_dataset_path="side_val.zarr.zip")
    with initialize_config_dir(version_base=None, config_dir=str(config_dir())):
        composed = compose(cfg.config_name, overrides=train_overrides(cfg, str(tmp_path)))
    OmegaConf.resolve(composed)

    assert composed.task.shape_meta.obs.bbox.bbox_mode == "mask_channel"
    assert "bbox" in composed.task.dataset.shape_meta.obs
    assert composed.policy.obs_encoder.feature_aggregation == "box_pool"
    assert composed.training.num_epochs == 50
    assert composed.task.dataset.val_dataset_path == "side_val.zarr.zip"


def test_missing_dataset_path_does_not_start_training(tmp_path, monkeypatch):
    def fail(*_a, **_k):
        raise AssertionError("training should not start")

    monkeypatch.setattr("solarian_umi_skill.training.run_training", fail)
    with pytest.raises(ValueError, match="dataset_path"):
        UmiTrainer(_cfg(dataset_path=None), output_dir=str(tmp_path)).train()
