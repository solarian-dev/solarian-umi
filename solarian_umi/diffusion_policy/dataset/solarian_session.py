"""A solarian session as a UMI training set.

``solarian umi-slam`` writes the session layout of
:mod:`solarian.core.recording.session` into one ``.solarian.zip``, entries at
the archive root as in a ``.zarr.zip``: one parquet table and one mp4 per
camera per episode, the mp4 being the remapped GoPro-geometry footage at full
resolution. Two UMI-specific sidecars sit beside them::

    session.json, manifest.jsonl
    episodes/000000/
        episode.json             + "umi": {"robot0": {"demo_start_pose", "demo_end_pose"}}
        episode.parquet          obs.robot0.{x,y,z,rx,ry,rz,gripper_width}
        video/camera0.rgb.mp4
        tags/camera0.rgb.json    per frame, {tag_id: 4x2 corners}, inpainted at load
        targets/camera0.rgb.npz  optional: bbox (T, 4+), point (T, 2), mask, in video pixels

The same layout unzipped, as a directory, reads the same way. Training decodes
each episode once into the same replay buffer a ``.zarr.zip`` gives, at the
policy resolution, so the sampler is unchanged.
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import multiprocessing
import os
import re
import struct
import threading
import time
import zipfile
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import zarr

from solarian.core.recording.session import (
    EPISODE_FILE,
    EPISODES_DIR,
    MANIFEST_FILE,
    SESSION_FILE,
    TABLE_FILE,
    VIDEO_DIR,
)
from solarian_umi.diffusion_policy.codecs.imagecodecs_numcodecs import (
    JpegXl,
    register_codecs,
)
from solarian_umi.diffusion_policy.common.replay_buffer import ReplayBuffer
from solarian_umi.slam.support.policy_frame import PolicyFrame

register_codecs(verbose=False)

SUFFIX = ".solarian.zip"
#: Seconds between progress lines while a session decodes.
PROGRESS_INTERVAL_S = 10.0
TAGS_DIR = "tags"
TARGETS_DIR = "targets"
#: Per-frame vectors in source pixels, mapped to policy pixels by
#: :class:`PolicyFrame`. Training observes each under its own key, so a
#: shape_meta target is named after its array.
LOW_DIM_TARGETS = ("bbox", "point")

#: Gripper TCP position (m) and axis-angle rotation (rad) in the SLAM tag frame.
POS_FIELDS = ("x", "y", "z")
ROT_FIELDS = ("rx", "ry", "rz")
WIDTH_FIELD = "gripper_width"

#: Already compressed, so stored as they are: an mp4 member can then be
#: decoded straight out of the archive without inflating it first.
_STORED_SUFFIXES = (".mp4", ".parquet", ".npz")

_EPISODE_JSON = re.compile(rf"^{EPISODES_DIR}/(\d+)/{re.escape(EPISODE_FILE)}$")


def gripper_module(index: int) -> str:
    return f"robot{index}"


def camera_name(index: int) -> str:
    return f"camera{index}.rgb"


def episode_name(index: int) -> str:
    """An episode's path inside the session, e.g. ``episodes/000000``."""
    return f"{EPISODES_DIR}/{index:06d}"


def video_member(episode: str, camera: str) -> str:
    return f"{episode}/{VIDEO_DIR}/{camera}.mp4"


def tags_member(episode: str, camera: str) -> str:
    return f"{episode}/{TAGS_DIR}/{camera}.json"


def targets_member(episode: str, camera: str) -> str:
    return f"{episode}/{TARGETS_DIR}/{camera}.npz"


class _Window(io.RawIOBase):
    """``size`` bytes of ``raw`` from ``start``: seekable without reading what it skips."""

    def __init__(self, raw, start: int, size: int):
        self._raw, self._start, self._size, self._pos = raw, start, size, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = min(max(base + offset, 0), self._size)
        return self._pos

    def readinto(self, buffer) -> int:
        self._raw.seek(self._start + self._pos)
        data = self._raw.read(min(len(buffer), self._size - self._pos))
        buffer[:len(data)] = data
        self._pos += len(data)
        return len(data)


class SessionArchive:
    """A ``.solarian.zip`` or a session directory, read by member path."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self._zip = None if self.path.is_dir() else zipfile.ZipFile(self.path)
        self._names = set(self._zip.namelist()) if self._zip is not None else None
        # Opened once: on a network volume each open costs a round trip.
        self._raw = None

    def __enter__(self) -> "SessionArchive":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
        if self._raw is not None:
            self._raw.close()

    def exists(self, member: str) -> bool:
        if self._zip is not None:
            return member in self._names
        return (self.path / member).is_file()

    def open(self, member: str):
        """A seekable binary file for ``member``.

        A zip member's own ``seek`` reads what it skips, and the video decoder
        seeks, so a stored member is read through a window on the archive file.
        """
        if self._zip is None:
            return open(self.path / member, "rb")
        info = self._zip.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            return self._zip.open(member)
        if self._raw is None:
            self._raw = open(self.path, "rb")
        self._raw.seek(info.header_offset)
        name_len, extra_len = struct.unpack("<HH", self._raw.read(30)[26:30])
        window = _Window(self._raw, info.header_offset + 30 + name_len + extra_len, info.file_size)
        return io.BufferedReader(window)

    def read_text(self, member: str) -> str:
        with self.open(member) as fh:
            return fh.read().decode("utf-8")

    def npz_keys(self, member: str) -> set[str]:
        """The arrays an ``.npz`` member holds, from its directory alone.

        A targets file is mostly mask, and is read over a network volume.
        """
        if self._zip is not None and self._zip.getinfo(member).compress_type != zipfile.ZIP_STORED:
            with self.open(member) as fh, np.load(io.BytesIO(fh.read())) as npz:
                return set(npz.files)
        with self.open(member) as fh, np.load(fh) as npz:
            return set(npz.files)

    def episodes(self) -> list[str]:
        """Sealed ``ok`` episodes in index order, as member prefixes."""
        if self._zip is not None:
            indices = [int(m.group(1)) for m in map(_EPISODE_JSON.match, self._names) if m]
        else:
            root = self.path / EPISODES_DIR
            indices = [
                int(p.name) for p in root.iterdir()
                if p.name.isdigit() and (p / EPISODE_FILE).is_file()
            ] if root.is_dir() else []
        found = []
        for index in sorted(indices):
            name = episode_name(index)
            meta = json.loads(self.read_text(f"{name}/{EPISODE_FILE}"))
            if meta.get("status") == "ok":
                found.append(name)
        return found


def is_session(path: str | Path) -> bool:
    """A session directory, or a zip with ``session.json`` at its root."""
    path = Path(path).expanduser()
    if path.is_dir():
        return (path / SESSION_FILE).is_file()
    if not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as zf:
        return SESSION_FILE in zf.namelist()


def session_stamp(path: str | Path) -> float:
    """Changes whenever an episode is sealed, so it can key a cache."""
    path = Path(path).expanduser()
    if path.is_file():
        return path.stat().st_mtime
    manifest = path / MANIFEST_FILE
    return (manifest if manifest.is_file() else path / SESSION_FILE).stat().st_mtime


def zip_session(directory: str | Path, output: str | Path) -> Path:
    """Pack a session directory into ``output``, its entries at the archive root."""
    directory, output = Path(directory), Path(output)
    tmp = output.with_name(output.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as zf:
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            stored = path.suffix in _STORED_SUFFIXES
            zf.write(path, path.relative_to(directory).as_posix(),
                     compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)
    os.replace(tmp, output)
    return output


def write_tags(path: Path, frames: list[dict]) -> None:
    """``frames[i]`` maps tag id to its ``(4, 2)`` pixel corners in frame ``i``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([
        {str(tag): np.asarray(corners, dtype=np.float32).round(2).tolist()
         for tag, corners in frame.items()}
        for frame in frames
    ]))


def read_tags(archive: SessionArchive, member: str) -> list[list[np.ndarray]]:
    """Per frame, the corner arrays to inpaint. A missing member means no tags."""
    if not archive.exists(member):
        return []
    return [
        [np.asarray(corners, dtype=np.float32) for corners in frame.values()]
        for frame in json.loads(archive.read_text(member))
    ]


def write_targets(path: str | Path, episode: int, camera: str = camera_name(0),
                  mask: np.ndarray | None = None, **arrays: np.ndarray) -> None:
    """Add per-frame targets to a session: a 0/1 ``mask`` ``(T, H, W)`` and/or
    float arrays such as ``bbox`` ``(T, 4+)`` and ``point`` ``(T, 2)``.

    ``path`` is a ``.solarian.zip`` (the member is appended; one that exists
    already is refused, since a zip cannot replace a member in place) or a
    session directory.
    """
    arrays = {key: np.asarray(value, dtype=np.float32) for key, value in arrays.items()}
    if mask is not None:
        mask = np.asarray(mask, dtype=np.uint8)
        # Bit-packed: a full-resolution binary mask is 8x smaller before zlib.
        arrays["mask"] = np.packbits(mask.astype(bool), axis=-1)
        arrays["mask_width"] = np.array(mask.shape[-1])
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    member = targets_member(episode_name(episode), camera)
    path = Path(path).expanduser()
    if path.is_dir():
        (path / member).parent.mkdir(parents=True, exist_ok=True)
        (path / member).write_bytes(buffer.getvalue())
        return
    with zipfile.ZipFile(path, "a") as zf:
        if member in zf.namelist():
            raise FileExistsError(f"{member} is already in {path}")
        zf.writestr(member, buffer.getvalue(), compress_type=zipfile.ZIP_STORED)


def read_targets(archive: SessionArchive, member: str) -> dict[str, np.ndarray]:
    if not archive.exists(member):
        return {}
    with archive.open(member) as fh, np.load(io.BytesIO(fh.read())) as npz:
        out = {key: npz[key] for key in LOW_DIM_TARGETS if key in npz}
        if "mask" in npz:
            width = int(npz["mask_width"])
            out["mask"] = np.unpackbits(npz["mask"], axis=-1, count=width)
        return out


def episodes_missing_target(path: str | Path, key: str, camera: str = camera_name(0)) -> list[str]:
    """The episodes without ``key`` (``bbox``, ``point`` or ``mask``) for ``camera``."""
    with SessionArchive(path) as archive:
        return [
            episode for episode in archive.episodes()
            if not archive.exists(member := targets_member(episode, camera)) or key not in archive.npz_keys(member)
        ]


def has_target(path: str | Path, key: str, camera: str = camera_name(0)) -> bool:
    """Whether every episode carries ``key`` for ``camera``."""
    with SessionArchive(path) as archive:
        if not archive.episodes():
            return False
    return not episodes_missing_target(path, key, camera)


def load_session_replay_buffer(
    path: str | Path,
    out_res: tuple[int, int],
    store=None,
    no_mirror: bool = False,
    mirror_swap: bool = False,
    num_workers: int | None = None,
) -> ReplayBuffer:
    """Decode the session at ``path`` into a replay buffer at ``out_res`` ``(w, h)``.

    Holds the keys ``07_generate_replay_buffer.py`` writes, plus ``bbox``,
    ``point`` and ``mask`` when every episode has them for ``camera0.rgb``.
    """
    with SessionArchive(path) as archive:
        meta = json.loads(archive.read_text(SESSION_FILE))
        episodes = archive.episodes()
        if not episodes:
            raise ValueError(f"No sealed episodes in {archive.path}")
        modules = meta["robot"]["modules"]
        n_grippers = sum(1 for m in modules.values() if m["kind"] == "gripper")
        cameras = list(meta["cameras"])

        replay_buffer = ReplayBuffer.create_empty_zarr(
            storage=store if store is not None else zarr.MemoryStore())
        for episode in episodes:
            with archive.open(f"{episode}/{TABLE_FILE}") as fh:
                table = pq.read_table(fh)
            umi = json.loads(archive.read_text(f"{episode}/{EPISODE_FILE}"))["umi"]
            n = table.num_rows
            data = {}
            for i in range(n_grippers):
                module = gripper_module(i)
                data[f"{module}_eef_pos"] = _columns(table, module, POS_FIELDS)
                data[f"{module}_eef_rot_axis_angle"] = _columns(table, module, ROT_FIELDS)
                data[f"{module}_gripper_width"] = _columns(table, module, (WIDTH_FIELD,))
                for key in ("demo_start_pose", "demo_end_pose"):
                    data[f"{module}_{key}"] = np.repeat(
                        np.asarray(umi[module][key], dtype=np.float64)[None], n, axis=0)
            replay_buffer.add_episode(data=data, compressors=None)

        ow, oh = out_res
        n_steps = replay_buffer.n_steps
        for cam_id in range(len(cameras)):
            replay_buffer.data.require_dataset(
                name=f"camera{cam_id}_rgb",
                shape=(n_steps, oh, ow, 3),
                chunks=(1, oh, ow, 3),
                compressor=JpegXl(level=99, numthreads=1),
                dtype=np.uint8,
            )

        first_camera = meta["cameras"][cameras[0]]
        policy_frame = PolicyFrame(
            in_res=(int(first_camera["width"]), int(first_camera["height"])),
            out_res=out_res,
            no_mirror=no_mirror,
            mirror_swap=mirror_swap,
        )
        ends = replay_buffer.episode_ends[:]
        starts = np.concatenate([[0], ends[:-1]]).astype(int)
        workers = num_workers or multiprocessing.cpu_count()
        name = Path(path).name
        print(f"Decoding {name}: {len(episodes)} episodes, {n_steps} frames, {len(cameras)} camera(s) "
              f"at {ow}x{oh} on {workers} threads", flush=True)
        started = time.monotonic()
        members = [targets_member(e, cameras[0]) for e in episodes]
        # Targets in policy pixels, only when every episode has them.
        target_keys = [key for key in (*LOW_DIM_TARGETS, "mask")
                       if all(archive.exists(m) and key in archive.npz_keys(m) for m in members)]
        low_dim_keys = [key for key in target_keys if key in LOW_DIM_TARGETS]
        if "mask" in target_keys:
            replay_buffer.data.require_dataset(
                name="mask",
                shape=(n_steps, oh, ow),
                chunks=(1, oh, ow),
                compressor=ReplayBuffer.resolve_compressor("disk"),
                dtype=np.uint8,
            )

    # One archive per thread: a ZipFile's members share its file position, and
    # opening one on a network volume takes seconds.
    local = threading.local()
    thread_archives: list[SessionArchive] = []

    def thread_archive() -> SessionArchive:
        if not hasattr(local, "archive"):
            local.archive = SessionArchive(path)
            thread_archives.append(local.archive)
        return local.archive

    def load_targets(member: str, start: int) -> dict[str, np.ndarray]:
        # Per episode: unpacked, a session's full-resolution masks are hundreds of GB.
        targets = read_targets(thread_archive(), member)
        if "mask" in target_keys:
            masks = replay_buffer.data["mask"]
            for k, mask in enumerate(targets["mask"]):
                masks[start + k] = policy_frame.mask(mask)
        to_policy = {"bbox": policy_frame.box, "point": policy_frame.point}
        return {key: to_policy[key](targets[key]) for key in low_dim_keys}

    def decode(episode: str, cam_id: int, start: int, n: int) -> None:
        camera = cameras[cam_id]
        img_array = replay_buffer.data[f"camera{cam_id}_rgb"]
        task_archive = thread_archive()
        tags = read_tags(task_archive, tags_member(episode, camera))
        with task_archive.open(video_member(episode, camera)) as fh, av.open(fh) as container:
            stream = container.streams.video[0]
            stream.thread_count = 1
            k = 0
            for frame in container.decode(stream):
                if k >= n:
                    break
                img = frame.to_ndarray(format="rgb24")
                img_array[start + k] = policy_frame(img, tags[k] if tags else [])
                k += 1
        if k != n:
            raise ValueError(f"{episode}: {camera}.mp4 has {k} frames, the table {n}")

    printed = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        target_futures = [
            executor.submit(load_targets, member, int(start))
            for member, start in zip(members, starts)
        ] if target_keys else []
        futures = [
            executor.submit(decode, episode, cam_id, int(start), int(end - start))
            for episode, start, end in zip(episodes, starts, ends)
            for cam_id in range(len(cameras))
        ]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if time.monotonic() - printed >= PROGRESS_INTERVAL_S and done < len(futures):
                printed = time.monotonic()
                print(f"Decoding {name}: {done}/{len(futures)} videos", flush=True)
        targets = [future.result() for future in target_futures]
    for task_archive in thread_archives:
        task_archive.close()
    for key in low_dim_keys:
        replay_buffer.data.array(name=key, data=np.concatenate([t[key] for t in targets]),
                                 compressor=None)
    print(f"Decoded {name} in {time.monotonic() - started:.0f}s", flush=True)
    return replay_buffer


def _columns(table, module: str, fields) -> np.ndarray:
    return np.stack(
        [table.column(f"obs.{module}.{f}").to_numpy() for f in fields],
        axis=-1).astype(np.float32)
