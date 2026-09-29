# solarian-umi-skill

Run a [UMI](https://umi-gripper.github.io/) diffusion policy. The policy, the module names, and the motion parameters live in a yaml in the project, not in the skill.

## Install

From a Solarian project:

```bash
solarian install solarian-umi-skill
```

That installs the package into the current Python environment, adds it to `pyproject.toml`, and copies `templates/umi.yaml` to `conf/umi.yaml` when that file is not already there. An unpublished checkout installs the same way from a path:

```bash
solarian install plugins/solarian-umi-skill
```

The policy server this skill calls is `solarian-policy-diffusion`. Install that too if you serve the checkpoint yourself.

## Use

`conf/umi.yaml` is one policy. Copy it for another (`conf/cup_arrange.yaml`, `conf/pick_lego.yaml`) and point each at its own checkpoint. Start the skill with that filename:

```python
from solarian_umi_skill import UMISkill

self.start_skill(UMISkill(params={"config": "cup_arrange.yaml"}))
```

`config` is a filename in `conf/`, or a path. `arm`, `gripper`, and `camera` come from the file; pass them in `params` only to override it. Any other key in `params` overrides the same key in the file.

The skill runs on the robot (`execution_place` is `ROBOT`), so the controller sends `START_SKILL` with name `umi` and those params. One `UMISkill` runs at a time; a different file restarts it.

Serve the checkpoint named in the file:

```bash
solarian run-policy umi --config cup_arrange.yaml
```

With no `--config`, `run-policy` reads `conf/umi.yaml`.
