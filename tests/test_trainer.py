"""UmiTrainer reads a training file: the native UMI config plus the Solarian keys.

Each file is composed with Hydra and validated as ``solarian train`` does. No
training runs.
"""

import os
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pydantic import ValidationError

from solarian.core.ai.system1.trainer_registry import load_train_config
from solarian_umi_skill.trainer import SOLARIAN_KEYS, UmiTrainer

TRAINING = Path(__file__).resolve().parents[1] / "solarian_umi_skill" / "templates" / "training"
TEMPLATES = sorted(p.stem for p in TRAINING.glob("*.yaml"))
CKPT = "/ckpts/pick-epoch_0170.ckpt"


def _compose(name: str = "umi", *overrides: str):
    with initialize_config_dir(version_base=None, config_dir=str(TRAINING)):
        return compose(config_name=name, overrides=list(overrides))


def _load(name: str = "umi", *overrides: str):
    return load_train_config(_compose(name, *overrides))


def _load_without(key: str, name: str = "umi"):
    # Hydra's ~key cannot delete a key whose value is null.
    cfg = OmegaConf.load(TRAINING / f"{name}.yaml")
    parent, _, leaf = key.rpartition(".")
    del (OmegaConf.select(cfg, parent) if parent else cfg)[leaf]
    return load_train_config(cfg)


def test_the_plugin_ships_the_three_training_templates():
    assert TEMPLATES == ["umi", "umi_pick", "umi_pick_mask"]


@pytest.mark.parametrize("name", TEMPLATES)
def test_each_template_is_a_complete_standalone_training_file(name):
    raw = yaml.safe_load((TRAINING / f"{name}.yaml").read_text())
    assert "defaults" not in raw
    assert "hydra" not in raw
    assert "???" not in (TRAINING / f"{name}.yaml").read_text()

    trainer_cls, cfg_dict, cfg = _load(name)

    assert trainer_cls is UmiTrainer
    assert cfg.trainer == "umi"
    assert cfg.output_dir is None and cfg.modal is None
    assert cfg.task.dataset_path == cfg.task.dataset_mix[0].path
    assert cfg_dict["task"]["dataset"]["dataset_path"] == cfg.task.dataset_path


def test_umi_config_is_the_file_without_the_solarian_keys():
    _cls, cfg_dict, cfg = _load()
    umi = cfg.umi_config()

    assert set(umi) == set(cfg_dict) - SOLARIAN_KEYS
    assert umi["_target_"] == cfg_dict["_target_"]
    assert umi["policy"] == cfg_dict["policy"]


def test_the_cup_template_has_no_target():
    _cls, _d, cfg = _load("umi")
    umi = cfg.umi_config()

    assert "bbox" not in umi["task"]["shape_meta"]["obs"]
    assert umi["bbox_mode"] == "none"
    assert umi["policy"]["obs_encoder"]["feature_aggregation"] == "attention_pool_2d"
    assert umi["training"]["num_epochs"] == 50


def test_the_pick_template_draws_the_box_and_reads_the_cls_token():
    _cls, _d, cfg = _load("umi_pick")
    umi = cfg.umi_config()

    assert umi["task"]["shape_meta"]["obs"]["bbox"]["bbox_mode"] == "mask_channel"
    assert umi["bbox_mode"] == "mask_channel"
    assert umi["policy"]["obs_encoder"]["feature_aggregation"] is None
    assert umi["training"]["num_epochs"] == 250
    assert umi["optimizer"]["lr"] == 3.0e-4


def test_the_mask_template_observes_the_mask_through_box_pool():
    _cls, _d, cfg = _load("umi_pick_mask")
    umi = cfg.umi_config()

    assert umi["task"]["shape_meta"]["obs"]["mask"]["type"] == "mask"
    assert "bbox" not in umi["task"]["shape_meta"]["obs"]
    assert umi["bbox_mode"] == "mask"
    assert umi["policy"]["obs_encoder"]["feature_aggregation"] == "box_pool"


def test_overrides_reach_the_umi_config():
    _cls, _d, cfg = _load("umi_pick", "training.num_epochs=3", "batch_size=16",
                          "policy.obs_encoder.feature_aggregation=box_pool", "job_name=pick_v2")
    umi = cfg.umi_config()

    assert cfg.job_name == "pick_v2"
    assert umi["training"]["num_epochs"] == 3
    assert umi["dataloader"]["batch_size"] == 16
    assert umi["policy"]["obs_encoder"]["feature_aggregation"] == "box_pool"


def test_several_datasets_become_the_mix():
    _cls, _d, cfg = _load("umi_pick", "task.dataset_mix=[{path:side.zarr.zip,weight:0.7},{path:pile.zarr.zip,weight:0.3}]")

    assert cfg.task.dataset_path == "side.zarr.zip"
    assert [(e.path, e.weight) for e in cfg.task.dataset_mix] == [("side.zarr.zip", 0.7), ("pile.zarr.zip", 0.3)]


def test_paths_are_expanded():
    _cls, _d, cfg = _load("umi_pick", "training.init_from_ckpt='~/ckpts/pick.ckpt'",
                          "task.dataset.val_dataset_path='~/data/side_val.zarr.zip'",
                          "task.dataset_mix=[{path:'~/data/side.zarr.zip',weight:1.0}]")
    umi = cfg.umi_config()
    home = os.path.expanduser("~")

    assert umi["training"]["init_from_ckpt"] == f"{home}/ckpts/pick.ckpt"
    assert umi["task"]["dataset"]["val_dataset_path"] == f"{home}/data/side_val.zarr.zip"
    assert umi["task"]["dataset_mix"][0]["path"] == f"{home}/data/side.zarr.zip"
    assert umi["task"]["dataset_path"] == f"{home}/data/side.zarr.zip"


@pytest.mark.parametrize("extra", ["+num_epochs=10", "+policy_name=umi", "+overrides=[]"])
def test_a_stray_top_level_key_is_rejected(extra):
    with pytest.raises(ValidationError, match="Extra inputs"):
        _load("umi", extra)


@pytest.mark.parametrize("key", ["job_name", "output_dir", "modal", "n_action_steps", "multi_run"])
def test_every_top_level_key_is_required(key):
    with pytest.raises(ValidationError, match=key):
        _load_without(key)


@pytest.mark.parametrize("key", ["task.dataset.val_dataset_path", "training.init_from_ckpt", "training.resume"])
def test_the_keys_solarian_reads_are_required(key):
    with pytest.raises(ValidationError, match=key.rsplit(".", 1)[1]):
        _load_without(key)


def test_an_empty_dataset_mix_is_rejected():
    with pytest.raises(ValidationError, match="dataset_mix"):
        _load("umi", "task.dataset_mix=[]", "task.dataset_path=x.zarr.zip")


def test_a_file_without_trainer_names_the_training_folder():
    with pytest.raises(ValueError, match="conf/training/<name>"):
        _load("umi", "~trainer")


def _no_training(monkeypatch, started=None):
    def run(umi_config, output_dir):
        if started is None:
            raise AssertionError("training should not start")
        started.append(umi_config)

    monkeypatch.setattr("solarian_umi_skill.training.run_training", run)


@pytest.mark.parametrize("override, message", [
    ("training.init_from_ckpt={missing}", "training.init_from_ckpt"),
    ("task.dataset.val_dataset_path={missing}", "val_dataset_path"),
])
def test_a_missing_input_file_does_not_start_training(tmp_path, override, message):
    dataset = tmp_path / "cup.zarr.zip"
    dataset.write_bytes(b"")
    _cls, _d, cfg = _load("umi", f"task.dataset_mix=[{{path:{dataset},weight:1.0}}]",
                          override.format(missing=tmp_path / "missing"))

    with pytest.raises(FileNotFoundError, match=message):
        UmiTrainer(cfg, output_dir=str(tmp_path)).check_inputs()


def test_a_missing_dataset_does_not_start_training(tmp_path):
    side = tmp_path / "side.zarr.zip"
    side.write_bytes(b"")
    _cls, _d, cfg = _load(
        "umi", f"task.dataset_mix=[{{path:{side},weight:1.0}},{{path:{tmp_path / 'pile.zarr.zip'},weight:1.0}}]")

    with pytest.raises(FileNotFoundError, match="pile.zarr.zip"):
        UmiTrainer(cfg, output_dir=str(tmp_path)).check_inputs()


def test_inputs_on_the_volume_are_left_to_the_run(tmp_path):
    dataset = tmp_path / "cup.zarr.zip"
    dataset.write_bytes(b"")
    _cls, _d, cfg = _load("umi", f"task.dataset_mix=[{{path:{dataset},weight:1.0}},{{path:/data/files/pile.zarr.zip,weight:1.0}}]",
                          "training.init_from_ckpt=/data/outputs/pick/checkpoints/latest.ckpt")
    trainer = UmiTrainer(cfg)

    trainer.check_inputs(skip=lambda path: path.startswith("/data/"))
    with pytest.raises(FileNotFoundError, match="init_from_ckpt"):
        trainer.check_inputs()


def test_resume_needs_the_run_latest_checkpoint(tmp_path, monkeypatch):
    started = []
    _no_training(monkeypatch, started)
    dataset = tmp_path / "cup.zarr.zip"
    dataset.write_bytes(b"")
    _cls, _d, cfg = _load("umi", "training.resume=true", f"task.dataset_mix=[{{path:{dataset},weight:1.0}}]")
    trainer = UmiTrainer(cfg, output_dir=str(tmp_path))

    with pytest.raises(FileNotFoundError, match="latest.ckpt"):
        trainer.train()

    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "latest.ckpt").write_bytes(b"")
    trainer.train()
    assert started[0]["training"]["resume"] is True
    assert "trainer" not in started[0]
