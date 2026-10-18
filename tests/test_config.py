"""The yaml named in params is the whole UMISkill config."""

from solarian_umi_skill import UMISkill


def test_module_names_come_from_the_config_file(tmp_path, monkeypatch):
    conf = tmp_path / "conf" / "skills"
    conf.mkdir(parents=True)
    (conf / "cup_arrange.yaml").write_text(
        "arm: right_arm\ngripper: right_gripper\ncamera: right_wrist_camera\nfrequency: 8\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SOLARIAN_CONF_PATH", str(conf.parent))
    monkeypatch.delenv("SOLARIAN_DEPLOYMENT", raising=False)

    skill = UMISkill(params={"config": "cup_arrange.yaml"})

    assert skill.params["arm"] == "right_arm"
    assert skill.params["gripper"] == "right_gripper"
    assert skill.commanded_module_names() == {"right_arm", "right_gripper"}
    assert skill.config["frequency"] == 8
    assert skill.registry_name() == "umi"


def test_params_override_the_file(tmp_path, monkeypatch):
    conf = tmp_path / "conf" / "skills"
    conf.mkdir(parents=True)
    (conf / "pick.yaml").write_text(
        "arm: right_arm\ngripper: right_gripper\nfrequency: 8\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SOLARIAN_CONF_PATH", str(conf.parent))

    skill = UMISkill(params={"config": "pick.yaml", "frequency": 4, "arm": "left_arm"})

    assert skill.params["arm"] == "left_arm"
    assert skill.config["frequency"] == 4
    assert skill.config["arm"] == "left_arm"
