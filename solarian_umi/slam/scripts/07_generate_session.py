"""SLAM output to a solarian session, the counterpart of 07_generate_replay_buffer.

Reads each input's ``dataset_plan.pkl`` and writes one session episode per
plan episode: the trajectory as ``episode.parquet``, the remapped video range
at full resolution as ``video/camera<i>.rgb.mp4``, and that range's ArUco
corners as ``tags/camera<i>.rgb.json``. Nothing is inpainted, masked or
resized here; training does that when it loads the session. The session is
packed into one ``.solarian.zip``, entries at its root.
"""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import os
import pathlib
import pickle
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction

import av
import click
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from solarian.core.recording.schema import CameraKey, Column, RecordingSchema
from solarian.core.recording.session import (
    TABLE_FILE,
    CameraMeta,
    EpisodeMeta,
    SessionMeta,
    SessionPaths,
    append_manifest,
    directory_size,
)
from solarian_umi.diffusion_policy.dataset.solarian_session import (
    POS_FIELDS,
    ROT_FIELDS,
    WIDTH_FIELD,
    camera_name,
    episode_name,
    gripper_module,
    tags_member,
    write_tags,
    zip_session,
)

#: Matches solarian's recording VideoWriter defaults.
VIDEO_OPTIONS = {"codec": "libx264", "crf": 20, "keyint": 30, "preset": "veryfast",
                 "pix_fmt": "yuv420p"}


@dataclass
class UmiEpisodeMeta(EpisodeMeta):
    #: Per gripper module, the demo's first and last valid TCP pose, which the
    #: policy's ``*_wrt_start`` observation is taken against.
    umi: dict = field(default_factory=dict)


def build_schema(n_grippers: int, n_cameras: int) -> RecordingSchema:
    columns = [
        Column(name=name, dtype="float64", data_class="TimestampData", unit="s")
        for name in ("t", "wall_ts")
    ]
    obs_fields = []
    for i in range(n_grippers):
        module = gripper_module(i)
        columns.append(Column(name=f"obs.{module}.ts", dtype="float64",
                              data_class="TimestampData", unit="s"))
        for name in POS_FIELDS:
            columns.append(Column(name=f"obs.{module}.{name}", dtype="float32",
                                  data_class="PosData", unit="m"))
        for name in ROT_FIELDS:
            columns.append(Column(name=f"obs.{module}.{name}", dtype="float32",
                                  data_class="PosData", unit="rad"))
        columns.append(Column(name=f"obs.{module}.{WIDTH_FIELD}", dtype="float32",
                              data_class="PosData", unit="m"))
        obs_fields.append((module, POS_FIELDS + ROT_FIELDS + (WIDTH_FIELD,)))
    cameras = []
    for i in range(n_cameras):
        module, cam_field = camera_name(i).split(".")
        camera = CameraKey(module=module, field=cam_field, sensor=module)
        cameras.append(camera)
        columns.append(Column(name=camera.frame_index_column, dtype="int32",
                              data_class="ImageData"))
        columns.append(Column(name=camera.capture_ts_column, dtype="float64",
                              data_class="ImageData", unit="s", component=module))
    return RecordingSchema(columns=tuple(columns), cameras=tuple(cameras),
                           obs_fields=tuple(obs_fields), act_fields=())


def robot_structure(n_grippers: int, n_cameras: int) -> dict:
    modules = {
        gripper_module(i): {
            "class": "UmiGripper",
            "kind": "gripper",
            "components": {},
            # TCP pose of the handheld gripper, solved by SLAM.
            "pose_frame": "umi_slam_tag",
        }
        for i in range(n_grippers)
    }
    for i in range(n_cameras):
        module = camera_name(i).split(".")[0]
        modules[module] = {"class": "UmiCamera", "kind": "camera", "components": {}}
    return {"class": "UmiDataset", "is_sim": False, "modules": modules}


def episode_table(schema: RecordingSchema, plan_episode: dict) -> pa.Table:
    timestamps = np.asarray(plan_episode["episode_timestamps"], dtype=np.float64)
    n = len(timestamps)
    columns = {"t": timestamps - timestamps[0], "wall_ts": timestamps}
    for i, gripper in enumerate(plan_episode["grippers"]):
        module = gripper_module(i)
        pose = np.asarray(gripper["tcp_pose"])
        columns[f"obs.{module}.ts"] = timestamps
        for k, name in enumerate(POS_FIELDS + ROT_FIELDS):
            columns[f"obs.{module}.{name}"] = pose[:, k]
        columns[f"obs.{module}.{WIDTH_FIELD}"] = np.asarray(gripper["gripper_width"])
    for camera in schema.cameras:
        columns[camera.frame_index_column] = np.arange(n)
        columns[camera.capture_ts_column] = timestamps
    arrow = {"float64": pa.float64(), "float32": pa.float32(), "int32": pa.int32()}
    return pa.table({
        c.name: pa.array(np.asarray(columns[c.name]), arrow[c.dtype])
        for c in schema.columns
    })


def column_stats(table: pa.Table, schema: RecordingSchema) -> dict:
    """Running sums, the form the recorder seals into ``episode.json``."""
    stats = {}
    for name in schema.numeric_columns():
        values = table.column(name).to_numpy().astype(np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        stats[name] = {"min": float(finite.min()), "max": float(finite.max()),
                       "sum": float(finite.sum()), "sumsq": float(np.square(finite).sum()),
                       "count": int(finite.size)}
    return stats


def open_encoder(path: pathlib.Path, width: int, height: int, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream(
        VIDEO_OPTIONS["codec"], rate=Fraction(fps).limit_denominator(10000))
    stream.width, stream.height = width, height
    stream.pix_fmt = VIDEO_OPTIONS["pix_fmt"]
    stream.options = {"crf": str(VIDEO_OPTIONS["crf"]), "preset": VIDEO_OPTIONS["preset"],
                      "g": str(VIDEO_OPTIONS["keyint"])}
    return container, stream


def cut_video(mp4_path: str, tasks: list[dict], fps: float) -> None:
    """Re-encode each task's ``[frame_start, frame_end)`` into its own mp4.

    One sequential decode per source video: episode boundaries are not
    keyframes, so a stream copy would start on an undecodable frame.
    """
    tag_detection = pickle.load(open(os.path.join(os.path.dirname(mp4_path),
                                                  "tag_detection.pkl"), "rb"))
    tasks = sorted(tasks, key=lambda x: x["frame_start"])
    current = 0
    container = stream = None
    with av.open(mp4_path) as reader:
        in_stream = reader.streams.video[0]
        in_stream.thread_count = 1
        for frame_idx, frame in enumerate(reader.decode(in_stream)):
            if current >= len(tasks):
                break
            task = tasks[current]
            if frame_idx < task["frame_start"]:
                continue
            if frame_idx == task["frame_start"]:
                container, stream = open_encoder(
                    task["video"], frame.width, frame.height, fps)
            img = frame.to_ndarray(format="rgb24")
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                container.mux(packet)
            if frame_idx + 1 == task["frame_end"]:
                for packet in stream.encode():
                    container.mux(packet)
                container.close()
                container = None
                write_tags(task["tags"], [
                    {tag: det["corners"]
                     for tag, det in tag_detection[i]["tag_dict"].items()}
                    for i in range(task["frame_start"], task["frame_end"])
                ])
                current += 1
    if container is not None:
        container.close()
    if current != len(tasks):
        raise ValueError(f"{mp4_path} ended before frame {tasks[current]['frame_end']}")


@click.command()
@click.argument("input", nargs=-1)
@click.option("-o", "--output", required=True, help="Session archive, <name>.solarian.zip")
@click.option("-n", "--num_workers", type=int, default=None)
def main(input, output, num_workers):
    output = pathlib.Path(os.path.expanduser(output)).absolute()
    if output.exists():
        click.confirm(f"Output {output} exists! Overwrite?", abort=True)
    num_workers = num_workers or multiprocessing.cpu_count()

    plans = []
    for ipath in input:
        ipath = pathlib.Path(os.path.expanduser(ipath)).absolute()
        plan_path = ipath.joinpath("dataset_plan.pkl")
        if not plan_path.is_file():
            print(f"Skipping {ipath.name}: no dataset_plan.pkl")
            continue
        for plan_episode in pickle.load(plan_path.open("rb")):
            plans.append((ipath.joinpath("demos"), plan_episode))
    if not plans:
        raise click.ClickException("No dataset_plan.pkl in any input.")

    output.parent.mkdir(parents=True, exist_ok=True)
    # Built beside the output, so packing it does not cross filesystems.
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".solarian-") as build:
        root = pathlib.Path(build) / "session"
        write_session(plans, root, num_workers)
        print(f"Packing {output}")
        zip_session(root, output)
    print(f"Done! {len(plans)} episodes in {output}")


def write_session(plans: list, root: pathlib.Path, num_workers: int) -> None:
    """The session directory for ``plans``, ``(demos_dir, plan_episode)`` pairs."""
    n_grippers = len(plans[0][1]["grippers"])
    n_cameras = len(plans[0][1]["cameras"])
    for _, plan_episode in plans:
        assert len(plan_episode["grippers"]) == n_grippers
        assert len(plan_episode["cameras"]) == n_cameras

    first_video = plans[0][0].joinpath(plans[0][1]["cameras"][0]["video_path"])
    with av.open(str(first_video)) as container:
        in_stream = container.streams.video[0]
        width, height = in_stream.width, in_stream.height
        fps = float(in_stream.average_rate)

    schema = build_schema(n_grippers, n_cameras)
    paths = SessionPaths.create(root)
    SessionMeta(
        robot=robot_structure(n_grippers, n_cameras),
        schema=schema,
        cameras={
            camera_name(i): CameraMeta(width=width, height=height, source_fps=fps)
            for i in range(n_cameras)
        },
        video=VIDEO_OPTIONS,
        fps=fps,
    ).write(paths)

    videos = defaultdict(list)
    metas = []
    for index, (demos_path, plan_episode) in enumerate(plans):
        episode_dir = paths.episode_dir(index)
        episode_dir.mkdir(parents=True, exist_ok=True)
        table = episode_table(schema, plan_episode)
        pq.write_table(table, episode_dir / TABLE_FILE, compression="zstd")
        n = table.num_rows
        for cam_id, camera in enumerate(plan_episode["cameras"]):
            start, end = camera["video_start_end"]
            assert end - start == n
            name = camera_name(cam_id)
            videos[str(demos_path.joinpath(camera["video_path"]))].append({
                "frame_start": start,
                "frame_end": end,
                "video": paths.video_path(index, name),
                "tags": root / tags_member(episode_name(index), name),
            })
        timestamps = plan_episode["episode_timestamps"]
        names = [camera_name(i) for i in range(n_cameras)]
        metas.append(UmiEpisodeMeta(
            index=index,
            start_ts=float(timestamps[0]),
            end_ts=float(timestamps[-1]) + 1.0 / fps,
            n_rows=n,
            n_frames={name: n for name in names},
            video_coverage={name: 1.0 for name in names},
            dropped_frames={name: 0 for name in names},
            measured_hz={gripper_module(i): round(fps, 1) for i in range(n_grippers)},
            stats=column_stats(table, schema),
            umi={
                gripper_module(i): {
                    "demo_start_pose": np.asarray(g["demo_start_pose"]).tolist(),
                    "demo_end_pose": np.asarray(g["demo_end_pose"]).tolist(),
                }
                for i, g in enumerate(plan_episode["grippers"])
            },
        ))

    print(f"{len(videos)} videos used in total!")
    with tqdm(total=len(videos)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(cut_video, mp4, tasks, fps)
                       for mp4, tasks in videos.items()]
            for future in concurrent.futures.as_completed(futures):
                future.result()
                pbar.update(1)

    # The manifest is written last: an episode listed there has its video.
    for meta in metas:
        meta.write(paths)
        append_manifest(paths, {
            "episode": meta.index,
            "sealed_at": time.time(),
            "status": meta.status,
            "bytes": directory_size(paths.episode_dir(meta.index)),
            "n_rows": meta.n_rows,
            "duration": round(meta.duration, 3),
            "transfer": {"state": "pending"},
            "upload": {"state": "pending"},
        })


if __name__ == "__main__":
    main()
