# solarian-umi-skill

Run and train [UMI](https://umi-gripper.github.io/) diffusion policies in Solarian. The policy, the module names, and the motion parameters live in a yaml in the project, not in the skill.

## Install

From a Solarian project:

```bash
solarian install solarian-umi-skill
```

That installs the package into the current Python environment, adds it to `pyproject.toml`, and copies `templates/umi.yaml` to `conf/umi.yaml` and `templates/train-umi.yaml` to `conf/train-umi.yaml` when those files are not already there. An unpublished checkout installs the same way from a path:

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

The skill runs on the robot by default; set `execution_place: server` in the file to run it on the relay. On the robot, the controller sends `START_SKILL` with name `umi` and those params. One `UMISkill` runs at a time; a different file restarts it.

Serve the checkpoint named in the file:

```bash
solarian run-policy umi --config cup_arrange.yaml
```

With no `--config`, `run-policy` reads `conf/umi.yaml`.

## Point and pick

`UMIPnPSkill` runs a UMI checkpoint trained with a target box (`obs.bbox` with `bbox_mode: mask_channel`): the box is drawn as a fourth mask channel over `camera0_rgb`. Its config is `conf/umi_pnp.yaml`, the same keys as `conf/umi.yaml` plus:

- `execution_place`: `server` or `robot`, from a required deployment key (`pnp_execution_place` in the template).
- `bbox_field`: the camera's calculated field holding the box in capture pixels, default `bbox`.

```python
from solarian_umi_skill import UMIPnPSkill

self.start_skill(UMIPnPSkill(params={"config": "umi_pnp.yaml"}))
```

The box usually comes from a tracker service such as `solarian-sam2-tracker`, seeded when the operator clicks the object (`self.start_service(camera, "bbox", point=...)`). Before a target is set the field is all zeros and the policy gets the whole frame as its box, the placeholder it was trained with. The skill maps the box through the same remap, resize and crop as the frame.

On the server, the skill sits beside the tracker and the robot only captures and executes. On the robot, the bbox field must also be computed on the robot (`execution_place: robot` in its conf); `setup` refuses a robot-hosted skill reading a server-computed box, because that field never reaches the robot. Point both confs at one deployment key so they cannot disagree.

## Train

`conf/train-umi.yaml` is the base for local training, the same way `conf/umi.yaml` is the base for a policy. Copy it per task and change `job_name` and `dataset_path`. `policy` stays `umi`.

```text
conf/train-cup-arrange.yaml    job_name: cup_arrange     dataset_path: datasets/cup.zarr.zip
conf/cup_arrange.yaml          policy_server.checkpoint: models/cup_arrange/checkpoints/latest.ckpt

conf/train-pick-object.yaml    job_name: pick_object
conf/pick_object.yaml          policy_server.checkpoint: models/pick_object/checkpoints/latest.ckpt
```

Training uses the UMI code copied into this package (MIT, copyright 2023 Columbia Artificial Intelligence and Robotics Lab, [universal_manipulation_interface](https://github.com/real-stanford/universal_manipulation_interface)). `config_name` selects the Hydra config shipped with the plugin (default `train_diffusion_unet_timm_umi_cup_wild`). Install the training libraries with `pip install "solarian-umi-skill[train]"`.

```bash
solarian train --conf conf/train-cup-arrange
```

Checkpoints are written to `models/<job_name>/checkpoints/latest.ckpt`. Set that path as `policy_server.checkpoint` in the skill yaml, then start the same skill:

```python
self.start_skill(UMISkill(params={"config": "cup_arrange.yaml"}))
```

### Point-and-pick training

`conf/train-umi-pick.yaml` (from `templates/train-umi-pick.yaml`) uses the same trainer with `config_name: train_diffusion_unet_timm_umi_pick_bbox`, which adds the target box as a mask channel. The dataset needs a per-frame `bbox` array.

- `init_from_ckpt`: fine-tune from a checkpoint's weights. The optimizer, LR schedule and epoch count start fresh, and the normalizer is recomputed from `dataset_path`.
- `resume`: continue an interrupted run from `models/<job_name>/checkpoints/latest.ckpt` at its next epoch. `num_epochs` stays the run's total.
- `val_dataset_path`: a separate `.zarr.zip` used as the whole validation set; every episode of `dataset_path` trains. Unset, a random `val_ratio` (5%) of `dataset_path`'s episodes validates.
- `bbox_conditioning`: `cls` conditions on the ViT summary token. `box_pool` also adds the mean of the patch tokens under the box and the box corners. `box_pool` widens the conditioning, so a `cls` checkpoint cannot initialize it.

Command-line values that contain `=` need inner quotes:

```bash
solarian train --conf conf/train-umi-pick dataset_path=datasets/pick.zarr.zip
solarian train --conf conf/train-umi-pick dataset_path=datasets/pick.zarr.zip \
    "init_from_ckpt='models/pick/epoch=0170.ckpt'" optimizer_lr=1e-4 num_epochs=80
```

## SLAM

`solarian umi-slam` reads a directory of GoPro or Lumi collector MP4s and writes `dataset.zarr.zip`. That file is `dataset_path` in the training yaml. It does not read a `solarian record` session.

```bash
pip install "solarian-umi-skill[slam]"
solarian umi-slam datasets/cup_session
# writes datasets/cup_session/dataset.zarr.zip

# Lumi: once per camera and lens, with the collector on its Stream video screen.
solarian umi-calibrate-lumi --serial LUMI07FC9785   # writes conf/cameras/LUMI07FC9785.json
solarian umi-slam datasets/lumi_session --camera lumi
```

`--camera gopro` (the default) uses the GoPro Max Lens yaml shipped in the ORB-SLAM3 image.

`--camera lumi` turns a Lumi session into GoPro footage before mapping, so the GoPro intrinsics, masks and SLAM settings apply, and a policy trained on it sees the same image geometry as one trained on GoPro data:

1. Each demo's video is remapped into GoPro Max Lens geometry at 1440x1080, black where the Lumi lens does not see. The recording stays beside it as `raw_video_native.mp4`. The camera's calibration comes from `<session>/camera_calibration.json`, else `conf/cameras/<serial>.json` in the directory the command runs from.
2. `imu_data.json` is built from the collector's `.imu.json` sidecars, which keep each sample's time. The GPMF track only has one payload per second.
3. The camera-to-IMU time offset of each demo and one camera-to-IMU rotation are measured by matching the video's rotation to the gyro. They are written to `<session>/camera_imu.json` and to `demos/mapping/lumi_slam.yaml`.
4. After batch SLAM, `trajectory_report.txt` lists each demo's lost fraction, extent and largest frame-to-frame jump. The pipeline stops before the dataset plan when most demos fail those limits. `python -m solarian_umi_skill.slam.trajectory_check <session>` prints the same report for an existing session.

`umi-calibrate-lumi` opens a fullscreen window with solarian's 9x6 checkerboard on the left and the Lumi's live view on the right, with the detected corners and a 4x4 grid over the frame. Hold the Lumi in front of the screen and press SPACE to keep a view. Keep going until every edge cell turns from red to green and there are 15 views, then press c. A detected board is painted over in the preview, because the lens also sees the preview and could detect a picture of the board in it. What stays the same while the camera moves (the gripper) is tinted blue, and edge cells it mostly covers are not required, so the camera can stay mounted on the collector. Needs ffmpeg. Saved frames stay in `.cache/lumi_calibration/<serial>_<time>/`; `umi-calibrate-lumi <that folder> --serial S` calibrates from them again.

`umi-calibrate-lumi --from-remap conf/calibrations/<camera>_gopro_remap.json --serial S --output PATH` writes a calibration from an existing checkerboard calibration instead.

`--slam-settings PATH` mounts another ORB-SLAM3 yaml instead.

ORB-SLAM3 runs in Docker image `chicheng/orb_slam3:latest`. GoPro IMU extraction runs in `chicheng/openicc:latest`. The Lumi remap needs `ffmpeg` on the path. If the image is already on this machine, it is used. If it is missing, the command asks before downloading it and running it here.

The camera serial is read from the GPMF track in the MP4, which both GoPro and Lumi write. A GoPro file's start time comes from `creation_time` plus the timecode. A Lumi file muxed without `creation_time` uses the timecode plus the file's modification time.

The pipeline and the training code are copied from [Universal Manipulation Interface](https://github.com/real-stanford/universal_manipulation_interface), copyright 2023 Columbia Artificial Intelligence and Robotics Lab, MIT license. The license text is `solarian_umi_skill/slam/LICENSE.umi` and `solarian_umi_skill/diffusion_policy/LICENSE.umi`.
