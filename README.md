# solarian-umi

Run and train [UMI](https://umi-gripper.github.io/) diffusion policies in Solarian. The policy, the module names, and the motion parameters live in a yaml in the project, not in the skill.

## Install

From a Solarian project:

```bash
solarian install solarian-umi
```

That installs the package into the current Python environment, adds it to `pyproject.toml`, and copies every file under `templates/` into the same folder of `conf/` when it is not already there: `conf/skills/umi.yaml`, `conf/skills/umi_target.yaml`, and the training files `conf/training/umi.yaml`, `umi_pick.yaml`, `umi_pick_mask.yaml` and `umi_place.yaml`. An unpublished checkout installs the same way from a path:

```bash
solarian install plugins/solarian-umi
```

The policy server this skill calls is `solarian-diffusion`. Install that too if you serve the checkpoint yourself.

## Use

`conf/skills/umi.yaml` is one policy. Copy it for another (`conf/skills/cup_arrange.yaml`, `conf/skills/pick_lego.yaml`) and point each at its own checkpoint. Start the skill with that filename:

```python
from solarian_umi import UMISkill

self.start_skill(UMISkill(params={"config": "cup_arrange.yaml"}))
```

`config` is a filename in `conf/skills/`, or a path. `arm`, `gripper`, and `camera` come from the file; pass them in `params` only to override it. Any other key in `params` overrides the same key in the file.

The skill runs on the robot by default; set `execution_place: server` in the file to run it on the relay. On the robot, the controller sends `START_SKILL` with name `umi` and those params. One `UMISkill` runs at a time; a different file restarts it.

Serve the checkpoint named in the file's `server:` block (the
`diffusion-policy` service, from `solarian-diffusion`):

```bash
solarian run-service cup_arrange
```

The skill reaches it at `policy_server_url`, sending `policy_server_token` when
the server asks for one (a policy hosted on Solarian Base). With
`server.managed: true` the skill launches the server itself.

## Targets: pick and place

A checkpoint can be trained with targets, each an extra input channel over `camera0_rgb` (see [`targets.py`](https://github.com/solarian-dev/solarian-umi/blob/main/solarian_umi/targets.py)):

- a box, drawn as a filled rectangle (`obs.bbox`, `bbox_mode: mask_channel`): pick what a SAM 2 tracker boxes;
- a mask, the object's segmentation (`obs.mask`, `type: mask`): pick a thin or diagonal object;
- a point, drawn as a `point_box_px` square (`obs.point`, `point_mode: mask_channel`): place the held object where a TAPNext point tracker says.

The same `UMISkill` runs them. Its yaml adds:

- `targets`: each target the checkpoint observes, mapped to the camera's calculated field holding it in capture pixels, e.g. `{bbox: bbox}`, `{mask: target_mask}` or `{point: target_point}`. The field must be a `BoundingBoxData`, `MaskData` or `PointData` to match; `setup` refuses a missing, extra or mismatched one.
- `execution_place`: `server` or `robot`, required with targets, from a deployment key (`umi_target_execution_place` in the template `conf/skills/umi_target.yaml`).

```python
self.start_skill(UMISkill(params={"config": "place_object.yaml"}))
```

A box or mask usually comes from `solarian-sam2`, a point from `solarian-tapnext`, each seeded when the operator clicks the wrist image (`self.start_service(camera, "target_point", point=...)`). Before then a box is all zeros and the policy gets the whole frame as its box, a mask is None and the policy gets an empty mask, and a point is all zeros and the policy gets `[-1, -1]`: each the dataset's "no target". The skill maps every target through the same remap, resize and crop as the frame; a point that the crop removes becomes `[-1, -1]`, and a mask is cleared where the gripper and mirrors are painted black.

On the server, the skill sits beside the trackers and the robot only captures and executes. On the robot, the target fields must also be computed on the robot (`execution_place: robot` in their confs); `setup` refuses a robot-hosted skill reading a server-computed field, because that field never reaches the robot. Point the skill and the trackers at one deployment key so they cannot disagree.

## Train

A training run is one file, `conf/training/<name>.yaml`. It is the UMI training config written as `python train.py --config-name` reads it in the [UMI repo](https://github.com/real-stanford/universal_manipulation_interface), with the task inlined under `task:` instead of a `defaults:` list, plus four Solarian keys at the top:

- `trainer: umi` selects this plugin's trainer.
- `job_name` names the run; checkpoints go to `models/<job_name>/` unless `output_dir` is set.
- `output_dir`: the run folder, or null for `models/<job_name>`.
- `modal`: `solarian modal-train` settings, or null for local training only.

Everything else is UMI's: the model (`policy`), `optimizer`, `training` (epochs, LR schedule, checkpoint cadence, `init_from_ckpt`, `resume`), `dataloader`, `logging`, `checkpoint` and the `task` with its datasets. Nothing is inherited from another file and no value is filled in by code, so the file shows every value the run uses. `${eval:...}` and `${parse_dataset_field:...}` work as in UMI: the trainer registers them before the file resolves. Every top-level key is required and an unknown one is rejected, so a typo fails before training starts.

`solarian install` seeds four files, copied from `templates/training/`:

- `conf/training/umi.yaml`: camera and proprioception only, no target; the UMI `train_diffusion_unet_timm_umi_cup_wild` run.
- `conf/training/umi_pick.yaml`: a ViT-L CLIP encoder and UNet conditioned on a target box drawn as a 4th image channel, on the CLS token; the UMI `train_diffusion_unet_timm_umi_pick_lego_pile_bbox` run.
- `conf/training/umi_pick_mask.yaml`: as `umi_pick`, conditioned on the object's mask through the CLS token and `box_pool`.
- `conf/training/umi_place.yaml`: as `umi_pick_mask`, conditioned on a place point instead, from the `track_point` dataset op.

Copy one per run and change `job_name`, `task.dataset_mix` and whatever else the run changes; `trainer` stays `umi`.

```text
conf/training/cup_arrange.yaml    job_name: cup_arrange    task.dataset_mix: [{path: datasets/cup.zarr.zip, weight: 1.0}]
conf/skills/cup_arrange.yaml      server.checkpoint: models/cup_arrange/checkpoints/latest.ckpt

conf/training/pick_object.yaml    job_name: pick_object
conf/skills/pick_object.yaml      server.checkpoint: models/pick_object/checkpoints/latest.ckpt
```

Training uses the UMI code copied into this package (MIT, copyright 2023 Columbia Artificial Intelligence and Robotics Lab, [universal_manipulation_interface](https://github.com/real-stanford/universal_manipulation_interface)). Install the training libraries with `pip install "solarian-umi[train]"`. `solarian modal-train` uploads every dataset and checkpoint the file names.

```bash
solarian train --conf conf/training/cup_arrange
```

Checkpoints are written to `models/<job_name>/checkpoints/latest.ckpt`. Set that path as `server.checkpoint` in the skill yaml, then start the same skill:

```python
self.start_skill(UMISkill(params={"config": "cup_arrange.yaml"}))
```

### The keys a run usually changes

- `task.dataset_mix`: a list of `{path, weight}`. Each sample comes from entry `i` with probability `weight_i / sum(weights)`, whatever the entry's size, so a small set can be emphasized without copying it. Validation comes from the first entry (or `task.dataset.val_dataset_path`); the normalizer is fit on all of them. `task.dataset_path` is `${task.dataset_mix[0].path}`.
- `task.dataset.val_dataset_path`: a separate dataset (`.zarr.zip` or `.solarian.zip`) used as the whole validation set; every episode of the datasets trains. null: a random `task.dataset.val_ratio` (5%) of the first dataset's episodes validates.
- `training.init_from_ckpt`: fine-tune from a checkpoint's weights. The optimizer, LR schedule and epoch count start fresh, and the normalizer is recomputed from the training data.
- `training.resume`: continue an interrupted run from `<output_dir>/checkpoints/latest.ckpt` at its next epoch. `training.num_epochs` stays the run's total.
- `task.samples_per_epoch`: samples drawn per epoch, by dataset weight. An epoch is `task.samples_per_epoch / batch_size` steps and only sets how often validation, sampling and checkpoints run; `training.num_epochs` of them is the run. null: as many as the datasets hold together.
- `policy.obs_encoder.feature_aggregation`: null conditions on the ViT summary (CLS) token. `box_pool` also adds the mean of the patch tokens under the target (weighted by the mask's shape for a mask target) and its bounding corners, which widens the conditioning. A CLS checkpoint in `training.init_from_ckpt` still initializes it: its conditioning weights move to the CLS and low-dim positions and the new columns start at zero, so the widened network computes the same function as the checkpoint until training moves those weights.
- The targets in `task.shape_meta.obs`: `umi_pick` observes the dataset's `bbox` array, drawn as a filled rectangle. `umi_pick_mask` (with `bbox_mode: mask`) observes its `mask` array, the object's segmentation, which marks a thin or diagonal object such as a cable where a box covers mostly background. `umi_place` observes its `point` array, drawn as a `point_box_px` square; negative means no point. Each entry reads the dataset array of its own name: in a `.solarian.zip` the `track` op writes `bbox` and `mask` and the `track_point` op writes `point`, into `episodes/*/targets/camera0.rgb.npz`; for a `.zarr.zip`, add a mask with solarian_base `scripts/zarr_add_masks.py`. Training refuses a dataset without every target it observes. Each target is one more ViT input channel, in shape_meta order, so a policy can observe a pick box and a place point together. A checkpoint with fewer targets starts one with more: `training.init_from_ckpt` pads the patch embedding and the `box_pool` conditioning with zeros for the new ones, so the widened network starts as the checkpoint.

Any key is overridden on the command line. Values that contain `=` or start with `~` need inner quotes:

```bash
solarian train --conf conf/training/umi_pick "task.dataset_mix=[{path:datasets/pick.zarr.zip,weight:1.0}]"
solarian train --conf conf/training/umi_pick "training.init_from_ckpt='models/pick/epoch=0170.ckpt'" \
    optimizer.lr=1e-4 training.num_epochs=80
```

## SLAM

`solarian umi-slam` reads a directory of GoPro or Lumi collector MP4s and writes a training dataset. That dataset is a `task.dataset_mix` entry in a training file. The input is not a `solarian record` session.

```bash
pip install "solarian-umi[slam]"
solarian umi-slam datasets/cup_session
# writes datasets/cup_session/dataset.solarian.zip
solarian umi-slam datasets/cup_session --format zarr
# writes datasets/cup_session/dataset.zarr.zip

# Lumi: once per camera and lens, with the collector on its Stream video screen.
solarian umi-calibrate-lumi --serial LUMI07FC9785   # writes conf/calibrations/cameras/LUMI07FC9785.json
solarian umi-slam datasets/lumi_session --camera lumi
# or the collector's packed dataset (solarian lumi pack); --camera lumi is implied
solarian umi-slam datasets/cups.lumi.zip
# writes datasets/cups.solarian.zip
```

A `.lumi.zip` is extracted into a temporary work directory (or `--work-dir DIR`), which is removed after a successful run and kept after a failed one. `--yes` downloads missing Docker images without asking, for unattended runs such as Solarian Base's **Generate Solarian Dataset**.

`--format solarian` (the default) writes a solarian session, the layout `solarian record` uses, packed into one `.solarian.zip` with its entries at the archive root, as in a `.zarr.zip`. Solarian Base accepts it as an upload, and unzipped it is a session directory `export-dataset` reads. Training reads the zip directly; the mp4s are stored without zip compression, so they decode straight out of the archive:

```
dataset.solarian.zip
  session.json, manifest.jsonl
  episodes/000000/
    episode.json             + "umi": demo start and end pose per gripper
    episode.parquet          obs.robot0.{x,y,z,rx,ry,rz,gripper_width}, one row per frame
    video/camera0.rgb.mp4    the remapped 1440x1080 video, not inpainted, masked or resized
    tags/camera0.rgb.json    ArUco corners per frame
    targets/camera0.rgb.npz  optional: bbox and/or mask per frame, in video pixels
```

The video keeps the full resolution, so tracking and segmentation (SAM 2) can run on it, and it is far smaller than 224x224 JPEG-XL frames. Training decodes each episode once when it starts: tags are inpainted, the gripper is masked and the frame is cropped to the policy resolution, the same steps `--format zarr` bakes into `dataset.zarr.zip`. `task.dataset.cache_dir` (`/tmp/solarian_umi_cache` in the shipped configs) holds the decoded buffer: the first GPU rank decodes into it under a file lock and the others copy the result, and later runs on the same machine reuse it until the session changes. It must be on local disk; `null` decodes in every process. Add `targets/` with `solarian_umi.diffusion_policy.dataset.solarian_session.write_targets(path, episode, bbox=..., mask=...)`, which appends to the zip; a `bbox` or `mask` present in every episode becomes the `bbox` or `mask` array a pick config trains on.

`--format zarr` writes the 224x224 `dataset.zarr.zip` instead.

A clip named `mapping.mp4` in the session directory is the mapping video, and one named `gripper_calibration.mp4` is the gripper calibration video (either case). Without them, the largest clip is the mapping video and the earliest remaining clip of each camera is its gripper calibration. A Lumi clip renamed this way gets the leftover `.imu.json` sidecar whose length best matches it. A clip that cannot be opened, such as one whose recorder lost power while saving it, moves with its sidecar to `<session>/unreadable_videos/` and the session goes on without it; an unreadable mapping or gripper calibration video stops the run.

`--camera gopro` (the default) uses the GoPro Max Lens yaml shipped in the ORB-SLAM3 image.

`--camera lumi` turns a Lumi session into GoPro footage before mapping, so the GoPro intrinsics, masks and SLAM settings apply, and a policy trained on it sees the same image geometry as one trained on GoPro data:

1. Each demo's video is remapped into GoPro Max Lens geometry at 1440x1080, black where the Lumi lens does not see. The recording stays beside it as `raw_video_native.mp4`. The camera's calibration comes from `<session>/camera_calibration.json`, else `conf/calibrations/cameras/<serial>.json` in the directory the command runs from.
2. `imu_data.json` is built from the collector's `.imu.json` sidecars, which keep each sample's time. The GPMF track only has one payload per second.
3. The camera-to-IMU time offset of each demo and one camera-to-IMU rotation are measured by matching the video's rotation to the gyro. They are written to `<session>/camera_imu.json` and to `demos/mapping/lumi_slam.yaml`, with the GoPro file's IMU noise: the Lumi's MPU-6050 reads about 3.5% over g when still, and trusting its accelerometer more made demos diverge when the camera turned to look straight down. So is the sensor's rolling-shutter readout, measured from how much later the gyro lines up with features lower in the frame (about 22 ms top to bottom on the IMX477). `--rolling-shutter` undoes it in the remap with the gyro's rotation over each frame's readout. It is off by default: on a 148-demo session it kept about ten more demos tracking through medium-speed turns, but about as many of those diverged, and the yield stayed within the spread between maps.
4. After batch SLAM, `trajectory_report.txt` lists each demo's lost fraction, extent and largest frame-to-frame jump. The pipeline stops before the dataset plan when most demos fail those limits. `python -m solarian_umi.slam.trajectory_check <session>` prints the same report for an existing session.

ORB-SLAM3 builds a different map each time it runs on the same mapping video, and how many demos track depends mostly on the map: on one 148-demo Lumi session, five maps let 2, 70, 94, 95 and 103 demos through. So the pipeline builds six maps side by side (`run_pipeline.py --map-candidates N`), tracks 24 demos spread over the session on each (`--probe-demos N`), and keeps the map most of them pass on. The other maps stay in `demos/mapping/map_candidates/`, and each candidate's score is in `<session>/map_selection.json`. Good maps are rare: on a 247-demo merge of two sessions, nine new maps passed 13 to 37 of 48 probe demos, against 42 for a map kept from an earlier run, which let 221 demos through where a fresh best of three had let 111. `--map-library DIR` (`solarian umi-slam --map-library DIR`) keeps the best map of each mapping take there. A later run on the same take, such as a merged dataset that shares it, scores that map alongside its new ones, keeps it on a tie, and replaces it only when a new map beats it. A map is tracked with the yaml it was built with, saved beside it as `map_slam_settings.yaml`, not with the session's new fit: on a map built with another camera-to-IMU rotation, 0.85° away cost a fifth of the passing demos and 2.5° nearly all of them. A `demos/mapping/map_atlas.osa` that already exists is used as it is, with the `map_slam_settings.yaml` beside it if there is one. Tracking on a map does not repeat either: of the demos that diverge on one run, most pass on another. So demos with no trajectory, or one that fails the limits, are tracked again on the kept map, up to twice (`--retries N`).

`03_batch_slam.py` ends with each demo that got no trajectory and why: the exit code or timeout, and ORB-SLAM3's last reason to stop, such as `Lost tracking on 60 >= 60 frames`. Its `slam_stdout.txt` stays in the demo's folder.

### Recording a session SLAM can track

A demo is lost when ORB-SLAM3 cannot place 60 of its frames (two seconds at 30 fps; they need not be in a row) or when its trajectory fails the limits above. On that same session, recorded with 8 ms exposure, demos dropped more often the faster the wrist turned, because a fast turn blurs the frames:

| Peak wrist rotation | Demos dropped |
| --- | --- |
| under 1.5 rad/s | 0% |
| 1.5 to 2.0 rad/s | 6% |
| 2.0 to 2.4 rad/s | 27% |
| 2.4 rad/s and over | 58% |

A session recorded with 4 ms exposure (`max_exposure_us: 4000` in the collector config) lost no demo to blur, the six that turned faster than 2.0 rad/s included. The table had plenty of texture: ORB-SLAM3 found as many features in the frames of failed demos as in those of tracked ones, looking straight down included.

- Record with a short exposure, 4 ms or less, and light the table so the image stays bright at that exposure.
- In the mapping video, cover every view the demos will have: look straight down on the work area from the heights the demos reach, as well as across the table. With a poor map, demos that looked down steeper than the mapping video did were the ones lost.
- Hold still for half a second at the start of each demo, where SLAM finds its place in the map.
- A plain, glossy or regularly patterned table (one colour, a sheet of glass, a grid of tiles) gives SLAM little to tell views apart by. Put tape or a printed mat on it before recording the mapping video and leave it in place for the whole session: the map holds only what the mapping video saw, so texture added or moved afterwards helps no demo find the map.

`umi-calibrate-lumi` opens a fullscreen window with solarian's 9x6 checkerboard on the left and the Lumi's live view on the right, with the detected corners and a 4x4 grid over the frame. Hold the Lumi in front of the screen and press SPACE to keep a view. Keep going until every edge cell turns from red to green and there are 15 views, then press c. A detected board is painted over in the preview, because the lens also sees the preview and could detect a picture of the board in it. What stays the same while the camera moves (the gripper) is tinted blue, and edge cells it mostly covers are not required, so the camera can stay mounted on the collector. Needs ffmpeg. Saved frames stay in `.cache/lumi_calibration/<serial>_<time>/`; `umi-calibrate-lumi <that folder> --serial S` calibrates from them again.

`umi-calibrate-lumi --from-remap conf/calibrations/<camera>_gopro_remap.json --serial S --output PATH` writes a calibration from an existing checkerboard calibration instead.

`--slam-settings PATH` mounts another ORB-SLAM3 yaml instead.

ORB-SLAM3 runs in Docker image `chicheng/orb_slam3:latest`. GoPro IMU extraction runs in `chicheng/openicc:latest`. With `--camera lumi`, ORB-SLAM3 runs in `solariandev/orb_slam3:lumi-inliers15` instead: the same image with the inlier floor for the first second after relocalization lowered from 30 to 15, built from `solarian_umi/slam/Dockerfile.lumi`. The Lumi remap needs `ffmpeg` on the path. If the image is already on this machine, it is used. If it is missing, the command asks before downloading it and running it here.

The camera serial is read from the GPMF track in the MP4, which both GoPro and Lumi write. A GoPro file's start time comes from `creation_time` plus the timecode. A Lumi file muxed without `creation_time` uses the timecode plus the file's modification time.

## License

[Apache License 2.0](https://github.com/solarian-dev/solarian-umi/blob/main/LICENSE). Copyright 2026 PrimaryKey Inc. See [NOTICE](https://github.com/solarian-dev/solarian-umi/blob/main/NOTICE).

The pipeline and the training code are copied from [Universal Manipulation Interface](https://github.com/real-stanford/universal_manipulation_interface), copyright 2023 Columbia Artificial Intelligence and Robotics Lab, MIT license. The license text is `solarian_umi/slam/LICENSE.umi` and `solarian_umi/diffusion_policy/LICENSE.umi`. This plugin is maintained independently of the UMI project and is not affiliated with, sponsored by, or endorsed by UMI, its authors, Stanford University, Columbia University or Toyota Research Institute.

ORB-SLAM3 is GPLv3. It runs inside the separate Docker images above and is not part of this package or covered by its license.

Solarian and the Solarian logo are trademarks of PrimaryKey Inc. See the [trademark policy](https://github.com/solarian-dev/solarian/blob/main/TRADEMARKS.md).
