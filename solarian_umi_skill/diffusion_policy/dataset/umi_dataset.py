# Adapted from Universal Manipulation Interface.
# Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab.
# https://github.com/real-stanford/universal_manipulation_interface
# Licensed under the MIT License. The full text is LICENSE.umi
# beside this package.

import copy
from typing import Dict, Optional

import os
from datetime import datetime
import pathlib
import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits
from tqdm import trange, tqdm
from filelock import FileLock
import shutil

from solarian_umi_skill.diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from solarian_umi_skill.diffusion_policy.common.normalize_util import (
    array_to_stats, concatenate_normalizer, get_identity_normalizer_from_stat,
    get_image_identity_normalizer, get_range_normalizer_from_stat)
from solarian_umi_skill.diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from solarian_umi_skill.diffusion_policy.common.pytorch_util import dict_apply
from solarian_umi_skill.diffusion_policy.common.replay_buffer import ReplayBuffer
from solarian_umi_skill.diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from solarian_umi_skill.diffusion_policy.dataset.base_dataset import BaseDataset
from solarian_umi_skill.diffusion_policy.dataset.solarian_session import (
    is_session, load_session_replay_buffer, session_stamp)
from solarian_umi_skill.diffusion_policy.model.common.normalizer import LinearNormalizer
from solarian_umi_skill.umi.common.pose_util import pose_to_mat, mat_to_pose10d

register_codecs()

def _try_load_rl_q_v(dataset_path: str):
    """Best-effort load of per-frame Q and V arrays from the source zarr.

    ``zarr_q_v.py`` writes ``rl/q`` and ``rl/v`` (shape ``(n_frames,)``,
    NaN where the frame is not a valid IQL anchor) alongside ``data/``
    and ``meta/`` in the source ``.zarr.zip``. ``ReplayBuffer.copy_from_store``
    only copies ``meta/`` and ``data/``, so the in-memory replay buffer
    used by the rest of this dataset never sees these arrays. We open
    the source zip-store directly, copy them into RAM (a few MB at
    most, since they're per-frame fp32 vectors), and return them so
    ``__getitem__`` can hand back per-sample advantages without touching
    the zarr again. Returns ``(None, None)`` if the source zarr has no
    ``rl/q`` / ``rl/v`` -- callers treat this as "no advantage info"
        and fall back to plain BC behavior.
    """
    if is_session(dataset_path):
        return None, None
    try:
        with zarr.ZipStore(dataset_path, mode='r') as zip_store:
            root = zarr.open(zip_store, mode='r')
            if 'rl' not in root:
                return None, None
            rl_grp = root['rl']
            if 'q' not in rl_grp or 'v' not in rl_grp:
                return None, None
            # Force full read into RAM. The arrays are 1-D float32 of
            # length n_frames (a few MB even for the largest UMI sets).
            q = np.asarray(rl_grp['q'][:], dtype=np.float32)
            v = np.asarray(rl_grp['v'][:], dtype=np.float32)
        if q.shape != v.shape or q.ndim != 1:
            # Defensive: if these ever go out of sync just disable AWR
            # rather than indexing into a mismatched array.
            return None, None
        return q, v
    except Exception:
        # Any I/O hiccup -> treat as "no advantage info". The caller
        # will silently skip the AWR weighting path.
        return None, None


def _policy_res(shape_meta: dict) -> tuple:
    """``(width, height)`` of the rgb observations, which must all agree."""
    sizes = {
        (attr['shape'][2], attr['shape'][1])
        for attr in shape_meta['obs'].values()
        if attr.get('type', 'low_dim') == 'rgb'
    }
    assert len(sizes) == 1, f"rgb observations disagree on resolution: {sizes}"
    return sizes.pop()


def open_replay_buffer(dataset_path: str, store, out_res: tuple) -> ReplayBuffer:
    """A ``.zarr.zip`` or a solarian session, copied into ``store``.

    A session is decoded to ``out_res`` ``(width, height)``; a zarr is
    already at its stored resolution.
    """
    if is_session(dataset_path):
        return load_session_replay_buffer(dataset_path, out_res, store=store)
    with zarr.ZipStore(dataset_path, mode='r') as zip_store:
        return ReplayBuffer.copy_from_store(src_store=zip_store, store=store)


def mix_sampler(lengths, weights, num_samples: Optional[int]) -> torch.utils.data.Sampler:
    """Sampler over a ConcatDataset of parts with ``lengths``.

    A sample comes from part ``i`` with probability ``weights[i] /
    sum(weights)`` whatever its length, so a small dataset can be drawn
    often without copying it. A single part is drawn without replacement,
    reshuffled on every pass. ``num_samples`` is the samples per epoch;
    None is the total length.
    """
    if num_samples is None:
        num_samples = int(sum(lengths))
    if len(lengths) == 1:
        return torch.utils.data.RandomSampler(range(lengths[0]), num_samples=num_samples)
    per_sample = torch.cat([
        torch.full((n,), w / sum(weights) / n, dtype=torch.double)
        for n, w in zip(lengths, weights)
    ])
    return torch.utils.data.WeightedRandomSampler(per_sample, num_samples, replacement=True)


class UmiDataset(BaseDataset):
    def __init__(self,
        shape_meta: dict,
        dataset_path: str,
        cache_dir: Optional[str]=None,
        pose_repr: dict={},
        action_padding: bool=False,
        temporally_independent_normalization: bool=False,
        repeat_frame_prob: float=0.0,
        seed: int=42,
        val_ratio: float=0.0,
        max_duration: Optional[float]=None,
        # Separate .zarr.zip or solarian session used as the whole validation set. When set,
        # val_ratio is ignored and every episode of dataset_path trains.
        val_dataset_path: Optional[str]=None,
        # PA-RL: optional per-frame cache of "optimized" action chunks
        # produced by a relabel pass (see
        # diffusion_policy.policy.parl_action_optimizer.optimize_actions
        # and the workspace's _parl_relabel hook). When present, each
        # __getitem__ emits two extra keys:
        #
        #   action_optimized:     (H, A) UNNORMALIZED action chunk that
        #                         the diffusion policy should distill
        #                         toward via the standard DDPM loss.
        #   has_action_optimized: scalar 1.0 / 0.0 gate (matches the
        #                         has_advantage convention).
        #
        # When None (the default), no extra keys are emitted and this
        # dataset is byte-identical to the legacy AWR / BC code paths.
        # Both arguments accept either a numpy memmap-able .npy path
        # OR an already-loaded numpy array (used by the workspace's
        # in-memory epoch-boundary swap).
        action_optimized_path: Optional[str]=None,
        action_optimized_valid_path: Optional[str]=None,
    ):
        self.pose_repr = pose_repr
        self.obs_pose_repr = self.pose_repr.get('obs_pose_repr', 'rel')
        self.action_pose_repr = self.pose_repr.get('action_pose_repr', 'rel')
        
        out_res = _policy_res(shape_meta)
        if cache_dir is None:
            replay_buffer = open_replay_buffer(dataset_path, zarr.MemoryStore(), out_res)
        else:
            # determine path name
            if is_session(dataset_path):
                mod_time = session_stamp(dataset_path)
            else:
                mod_time = os.path.getmtime(dataset_path)
            stamp = datetime.fromtimestamp(mod_time).isoformat()
            stem_name = os.path.basename(os.path.normpath(dataset_path)).split('.')[0]
            cache_name = '_'.join([stem_name, stamp])
            if is_session(dataset_path):
                # A session is decoded to the policy resolution, so that keys the cache too.
                cache_name += f'_{out_res[0]}x{out_res[1]}'
            cache_dir = pathlib.Path(os.path.expanduser(cache_dir))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir.joinpath(cache_name + '.zarr.mdb')
            lock_path = cache_dir.joinpath(cache_name + '.lock')
            
            # load cached file
            print('Acquiring lock on cache.')
            with FileLock(lock_path):
                # cache does not exist
                if not cache_path.exists():
                    try:
                        with zarr.LMDBStore(str(cache_path),     
                            writemap=True, metasync=False, sync=False, map_async=True, lock=False
                            ) as lmdb_store:
                            print(f"Copying data to {str(cache_path)}")
                            open_replay_buffer(dataset_path, lmdb_store, out_res)
                        print("Cache written to disk!")
                    except Exception as e:
                        shutil.rmtree(cache_path)
                        raise e
            
            # open read-only lmdb store
            store = zarr.LMDBStore(str(cache_path), readonly=True, lock=False)
            replay_buffer = ReplayBuffer.create_from_group(
                group=zarr.group(store)
            )
        
        self.num_robot = 0
        rgb_keys = list()
        mask_keys = list()
        lowdim_keys = list()
        key_horizon = dict()
        key_down_sample_steps = dict()
        key_latency_steps = dict()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            # solve obs type
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'mask':
                mask_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

            if key.endswith('eef_pos'):
                self.num_robot += 1

            # solve obs_horizon
            horizon = shape_meta['obs'][key]['horizon']
            key_horizon[key] = horizon

            # solve latency_steps
            latency_steps = shape_meta['obs'][key]['latency_steps']
            key_latency_steps[key] = latency_steps

            # solve down_sample_steps
            down_sample_steps = shape_meta['obs'][key]['down_sample_steps']
            key_down_sample_steps[key] = down_sample_steps

        # solve action
        key_horizon['action'] = shape_meta['action']['horizon']
        key_latency_steps['action'] = shape_meta['action']['latency_steps']
        key_down_sample_steps['action'] = shape_meta['action']['down_sample_steps']

        if val_dataset_path is not None:
            self.val_replay_buffer = open_replay_buffer(
                val_dataset_path, zarr.MemoryStore(), out_res)
            val_mask = np.zeros(replay_buffer.n_episodes, dtype=bool)
        else:
            self.val_replay_buffer = None
            val_mask = get_val_mask(
                n_episodes=replay_buffer.n_episodes, 
                val_ratio=val_ratio,
                seed=seed
            )
        self.val_dataset_path = val_dataset_path
        train_mask = ~val_mask

        self.sampler_lowdim_keys = list()
        for key in lowdim_keys:
            if not 'wrt' in key:
                self.sampler_lowdim_keys.append(key)
    
        for key in replay_buffer.keys():
            if key.endswith('_demo_start_pose') or key.endswith('_demo_end_pose'):
                self.sampler_lowdim_keys.append(key)
                query_key = key.split('_')[0] + '_eef_pos'
                key_horizon[key] = shape_meta['obs'][query_key]['horizon']
                key_latency_steps[key] = shape_meta['obs'][query_key]['latency_steps']
                key_down_sample_steps[key] = shape_meta['obs'][query_key]['down_sample_steps']

        # Masks are per-frame images to the sampler: read lazily from the
        # compressed store and skipped by ignore_rgb, so normalizer_data never
        # loads them.
        sampler = SequenceSampler(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            rgb_keys=rgb_keys + mask_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=key_horizon,
            key_latency_steps=key_latency_steps,
            key_down_sample_steps=key_down_sample_steps,
            episode_mask=train_mask,
            action_padding=action_padding,
            repeat_frame_prob=repeat_frame_prob,
            max_duration=max_duration
        )
        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.rgb_keys = rgb_keys
        self.mask_keys = mask_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.val_mask = val_mask
        self.action_padding = action_padding
        self.repeat_frame_prob = repeat_frame_prob
        self.max_duration = max_duration
        self.sampler = sampler
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False

        # PA-RL optimized-action cache. Configured here from the
        # ctor; the workspace also calls ``set_action_optimized`` to
        # swap in fresh in-memory arrays after each relabel pass. We
        # store the action shape parameters so ``set_action_optimized``
        # can sanity-check inputs without re-deriving them. n_frames
        # is the per-frame index space the cache must cover, matching
        # rl/q + rl/v written by zarr_q_v.py.
        self.action_horizon = int(key_horizon['action'])
        self.action_dim = int(shape_meta['action']['shape'][0])
        self.n_frames_total = int(replay_buffer.n_steps)
        self.action_optimized: Optional[np.ndarray] = None
        self.action_optimized_valid: Optional[np.ndarray] = None
        if action_optimized_path is not None:
            self._load_action_optimized_from_paths(
                action_optimized_path, action_optimized_valid_path,
            )

        # Best-effort per-frame Q/V (written by zarr_q_v.py). Always
        # loaded if available; the consuming policy decides whether to
        # use them via its own awr_beta knob -- so this is a strict
        # superset of the legacy BC dataset shape and never *requires*
        # rl/q + rl/v to be present.
        self.rl_q, self.rl_v = _try_load_rl_q_v(dataset_path)
        self.has_rl_qv = self.rl_q is not None and self.rl_v is not None
        if self.has_rl_qv:
            valid_mask = ~(np.isnan(self.rl_q) | np.isnan(self.rl_v))
            n_valid = int(valid_mask.sum())
            n_total = int(self.rl_q.shape[0])
            adv_valid = (self.rl_q[valid_mask] - self.rl_v[valid_mask]) \
                if n_valid > 0 else np.zeros(0, dtype=np.float32)
            print(
                f"[UmiDataset] loaded rl/q + rl/v from {dataset_path}: "
                f"{n_valid}/{n_total} frames have advantage "
                f"({100.0 * n_valid / max(1, n_total):.1f}%). "
                f"A: mean={adv_valid.mean() if n_valid else float('nan'):+.4f} "
                f"std={adv_valid.std() if n_valid else float('nan'):.4f} "
                f"min={adv_valid.min() if n_valid else float('nan'):+.4f} "
                f"max={adv_valid.max() if n_valid else float('nan'):+.4f}"
            )
        else:
            print(
                f"[UmiDataset] no rl/q + rl/v in {dataset_path}; "
                f"AWR weighting will be a no-op even if policy.awr_beta > 0."
            )

    
    def _load_action_optimized_from_paths(
        self,
        actions_path: str,
        valid_path: Optional[str],
    ) -> None:
        """Memory-map the per-frame PA-RL optimized-action cache.

        Args:
            actions_path: ``.npy`` of shape ``(n_frames_total, H, A)``
                ``float32``. Out-of-cache rows can hold any value;
                the valid mask is what gates them.
            valid_path: optional ``.npy`` of shape ``(n_frames_total,)``
                ``bool``. When ``None`` we assume every row is valid;
                this is only safe if the writer guarantees a value at
                every anchor index in the dataset.
        """
        actions = np.load(actions_path, mmap_mode='r')
        if actions.ndim != 3:
            raise ValueError(
                f"action_optimized array at {actions_path} has shape "
                f"{actions.shape}, expected 3D (n_frames, H, A)"
            )
        if actions.shape[1] != self.action_horizon \
                or actions.shape[2] != self.action_dim:
            raise ValueError(
                f"action_optimized chunk shape {actions.shape[1:]} != "
                f"({self.action_horizon}, {self.action_dim})"
            )
        if actions.shape[0] != self.n_frames_total:
            print(
                f"[UmiDataset] WARN: action_optimized n_frames "
                f"{actions.shape[0]} != dataset n_frames "
                f"{self.n_frames_total}; will gate by valid mask only."
            )
        if valid_path is not None:
            valid = np.load(valid_path, mmap_mode='r').astype(bool, copy=False)
            if valid.ndim != 1 or valid.shape[0] != actions.shape[0]:
                raise ValueError(
                    f"action_optimized_valid shape {valid.shape} "
                    f"!= ({actions.shape[0]},)"
                )
        else:
            # Caller asserted "every row valid". We still need a real
            # array so ``__getitem__`` can index uniformly.
            valid = np.ones(actions.shape[0], dtype=bool)
        self.action_optimized = actions
        self.action_optimized_valid = valid
        n_valid = int(valid.sum())
        print(
            f"[UmiDataset] loaded action_optimized cache from "
            f"{actions_path}: {n_valid}/{valid.shape[0]} valid frames "
            f"({100.0 * n_valid / max(1, valid.shape[0]):.1f}%)."
        )

    def set_action_optimized(
        self,
        actions: Optional[np.ndarray],
        valid: Optional[np.ndarray] = None,
    ) -> None:
        """In-memory swap of the PA-RL optimized-action cache.

        Used by ``TrainDiffusionUnetImageWorkspace`` to repoint the
        train + val datasets at a freshly-produced optimized-action
        array between epochs without rebuilding the dataset (which
        would re-pay the ZipStore -> ReplayBuffer copy).

        Pass ``actions=None`` to disable the cache (back to plain BC
        / AWR behavior). When ``valid is None`` we assume every row is
        valid (only safe if you wrote a value at every anchor index).
        """
        if self.replay_buffer is self.val_replay_buffer:
            # A separate validation zarr has its own frame indices, so the
            # training-frame cache does not apply; it keeps dataset actions.
            return
        if actions is None:
            self.action_optimized = None
            self.action_optimized_valid = None
            return
        if actions.ndim != 3 or actions.shape[1] != self.action_horizon \
                or actions.shape[2] != self.action_dim:
            raise ValueError(
                f"set_action_optimized: actions shape {actions.shape} "
                f"!= (*, {self.action_horizon}, {self.action_dim})"
            )
        if valid is None:
            valid = np.ones(actions.shape[0], dtype=bool)
        else:
            valid = valid.astype(bool, copy=False)
            if valid.shape != (actions.shape[0],):
                raise ValueError(
                    f"set_action_optimized: valid shape {valid.shape} "
                    f"!= ({actions.shape[0]},)"
                )
        self.action_optimized = actions
        self.action_optimized_valid = valid

    @property
    def has_action_optimized_cache(self) -> bool:
        return self.action_optimized is not None

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        if self.val_replay_buffer is not None:
            replay_buffer = self.val_replay_buffer
            episode_mask = np.ones(replay_buffer.n_episodes, dtype=bool)
            # Frame-indexed side data (Q/V, PA-RL cache) must match this zarr.
            val_set.replay_buffer = replay_buffer
            val_set.n_frames_total = int(replay_buffer.n_steps)
            val_set.rl_q, val_set.rl_v = _try_load_rl_q_v(self.val_dataset_path)
            val_set.has_rl_qv = val_set.rl_q is not None and val_set.rl_v is not None
            val_set.action_optimized = None
            val_set.action_optimized_valid = None
        else:
            replay_buffer = self.replay_buffer
            episode_mask = self.val_mask
        val_set.sampler = SequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=replay_buffer,
            rgb_keys=self.rgb_keys + self.mask_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=self.key_horizon,
            key_latency_steps=self.key_latency_steps,
            key_down_sample_steps=self.key_down_sample_steps,
            episode_mask=episode_mask,
            action_padding=self.action_padding,
            repeat_frame_prob=self.repeat_frame_prob,
            max_duration=self.max_duration
        )
        val_set.val_mask = ~episode_mask
        return val_set
    
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        return self.fit_normalizer(self.normalizer_data())

    def normalizer_data(self) -> Dict[str, np.ndarray]:
        """The low-dim obs and actions of every training sample, ``(N, T, D)``.

        Kept apart from :meth:`fit_normalizer` so several datasets can be
        concatenated and fit as one.
        """
        data_cache = {key: list() for key in self.lowdim_keys + ['action']}
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=32,
        )
        for batch in tqdm(dataloader, desc='iterating dataset to get normalization'):
            for key in self.lowdim_keys:
                data_cache[key].append(copy.deepcopy(batch['obs'][key]))
            data_cache['action'].append(copy.deepcopy(batch['action']))
        self.sampler.ignore_rgb(False)

        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key])
            assert data_cache[key].shape[0] == len(self.sampler)
            assert len(data_cache[key].shape) == 3
        return data_cache

    def fit_normalizer(self, data: Dict[str, np.ndarray]) -> LinearNormalizer:
        """Normalizer for ``data`` shaped like :meth:`normalizer_data`."""
        normalizer = LinearNormalizer()
        data_cache = dict()
        for key, value in data.items():
            B, T, D = value.shape
            data_cache[key] = value if self.temporally_independent_normalization \
                else value.reshape(B*T, D)

        # action
        assert data_cache['action'].shape[-1] % self.num_robot == 0
        dim_a = data_cache['action'].shape[-1] // self.num_robot
        action_normalizers = list()
        for i in range(self.num_robot):
            action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., i * dim_a: i * dim_a + 3])))              # pos
            action_normalizers.append(get_identity_normalizer_from_stat(array_to_stats(data_cache['action'][..., i * dim_a + 3: (i + 1) * dim_a - 1]))) # rot
            action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., (i + 1) * dim_a - 1: (i + 1) * dim_a])))  # gripper

        normalizer['action'] = concatenate_normalizer(action_normalizers)

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(data_cache[key])

            if key.endswith('pos') or 'pos_wrt' in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('pos_abs'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('rot_axis_angle') or 'rot_axis_angle_wrt' in key:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('gripper_width'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('bbox'):
                # Pixel-space bbox [x0, y0, x1, y1]. Two modes:
                #   * low_dim:      map observed range -> [-1, 1] (the bbox is
                #                   concatenated as 4 scalars onto the obs-
                #                   encoder feature vector, so a [-1, 1] scale
                #                   matches the proprio features).
                #   * mask_channel: keep raw pixel coords (identity) because
                #                   the encoder uses them to rasterize a binary
                #                   mask in pixel space. Normalizing to [-1, 1]
                #                   would force the encoder to invert the
                #                   normalization before rasterizing.
                bbox_attr = self.shape_meta['obs'].get(key, {})
                bbox_mode = bbox_attr.get('bbox_mode', 'low_dim')
                if bbox_mode == 'mask_channel':
                    this_normalizer = get_identity_normalizer_from_stat(stat)
                else:
                    this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys + self.mask_keys:
            normalizer[key] = get_image_identity_normalizer()
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
        # Resolve the dataset frame this sample anchors at BEFORE we
        # delegate to the sampler. The sampler stores
        # ``indices = [(current_idx, start_idx, end_idx, ...)]`` and
        # ``current_idx`` is the global frame index at which the action
        # chunk starts -- exactly the frame whose Q/V we need.
        try:
            anchor_idx = int(self.sampler.indices[idx][0])
        except Exception:
            anchor_idx = -1
        data = self.sampler.sample_sequence(idx)

        obs_dict = dict()
        for key in self.rgb_keys:
            if not key in data:
                continue
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.
            # T,C,H,W
            del data[key]
        for key in self.mask_keys:
            # Absent while the normalizer is fit: the sampler skips image keys.
            if key not in data:
                continue
            # T,H,W of 0/1 -> T,1,H,W
            obs_dict[key] = data.pop(key)[:, None].astype(np.float32)
        for key in self.sampler_lowdim_keys:
            obs_dict[key] = data[key].astype(np.float32)
            del data[key]
        
        # generate relative pose between two ees
        for robot_id in range(self.num_robot):
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([
                obs_dict[f'robot{robot_id}_eef_pos'],
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            for other_robot_id in range(self.num_robot):
                if robot_id == other_robot_id:
                    continue
                if not f'robot{robot_id}_eef_pos_wrt{other_robot_id}' in self.lowdim_keys:
                    continue
                other_pose_mat = pose_to_mat(np.concatenate([
                    obs_dict[f'robot{other_robot_id}_eef_pos'],
                    obs_dict[f'robot{other_robot_id}_eef_rot_axis_angle']
                ], axis=-1))
                rel_obs_pose_mat = convert_pose_mat_rep(
                    pose_mat,
                    base_pose_mat=other_pose_mat[-1],
                    pose_rep='relative',
                    backward=False)
                rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
                obs_dict[f'robot{robot_id}_eef_pos_wrt{other_robot_id}'] = rel_obs_pose[:,:3]
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}'] = rel_obs_pose[:,3:]
                
        # generate relative pose with respect to episode start
        for robot_id in range(self.num_robot):
            # HACK: add noise to episode start pose
            if (f'robot{other_robot_id}_eef_pos_wrt_start' not in self.shape_meta['obs']) and \
                (f'robot{other_robot_id}_eef_rot_axis_angle_wrt_start' not in self.shape_meta['obs']):
                continue
            
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([
                obs_dict[f'robot{robot_id}_eef_pos'],
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            
            # get start pose
            start_pose = obs_dict[f'robot{robot_id}_demo_start_pose'][0]
            # HACK: add noise to episode start pose
            start_pose += np.random.normal(scale=[0.05,0.05,0.05,0.05,0.05,0.05],size=start_pose.shape)
            start_pose_mat = pose_to_mat(start_pose)
            rel_obs_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=start_pose_mat,
                pose_rep='relative',
                backward=False)
            
            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            # HACK: add noise to episode start pose
            # obs_dict[f'robot{robot_id}_eef_pos_wrt_start'] = rel_obs_pose[:,:3]
            obs_dict[f'robot{robot_id}_eef_rot_axis_angle_wrt_start'] = rel_obs_pose[:,3:]

        del_keys = list()
        for key in obs_dict:
            if key.endswith('_demo_start_pose') or key.endswith('_demo_end_pose'):
                del_keys.append(key)
        for key in del_keys:
            del obs_dict[key]

        actions = list()
        for robot_id in range(self.num_robot):
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([
                obs_dict[f'robot{robot_id}_eef_pos'],
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            action_mat = pose_to_mat(data['action'][...,7 * robot_id: 7 * robot_id + 6])
            
            # solve relative obs
            obs_pose_mat = convert_pose_mat_rep(
                pose_mat, 
                base_pose_mat=pose_mat[-1],
                pose_rep=self.obs_pose_repr,
                backward=False)
            action_pose_mat = convert_pose_mat_rep(
                action_mat, 
                base_pose_mat=pose_mat[-1],
                pose_rep=self.obs_pose_repr,
                backward=False)
        
            # convert pose to pos + rot6d representation
            obs_pose = mat_to_pose10d(obs_pose_mat)
            action_pose = mat_to_pose10d(action_pose_mat)
        
            action_gripper = data['action'][..., 7 * robot_id + 6: 7 * robot_id + 7]
            actions.append(np.concatenate([action_pose, action_gripper], axis=-1))

            # generate data
            obs_dict[f'robot{robot_id}_eef_pos'] = obs_pose[:,:3]
            obs_dict[f'robot{robot_id}_eef_rot_axis_angle'] = obs_pose[:,3:]
            
        data['action'] = np.concatenate(actions, axis=-1)
        
        # Per-sample advantage for AWR-style finetuning. Always emitted
        # so the batch shape is identical regardless of whether the
        # source zarr carries rl/q + rl/v -- the policy decides whether
        # to use them. ``has_advantage`` is the gate the policy reads:
        # 1.0 means "this sample has a real Q-V signal, use it";
        # 0.0 means "fall back to plain BC weight=1 for this sample".
        # Frames flagged by zarr_q_v.py as not-an-IQL-anchor (NaN in
        # rl/q or rl/v) are degraded to has_advantage=0 here so the
        # downstream loss never sees a NaN.
        if self.has_rl_qv and 0 <= anchor_idx < int(self.rl_q.shape[0]):
            q = float(self.rl_q[anchor_idx])
            v = float(self.rl_v[anchor_idx])
            if np.isnan(q) or np.isnan(v):
                advantage = 0.0
                has_advantage = 0.0
            else:
                advantage = q - v
                has_advantage = 1.0
        else:
            advantage = 0.0
            has_advantage = 0.0

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32)),
            'advantage': torch.tensor(advantage, dtype=torch.float32),
            'has_advantage': torch.tensor(has_advantage, dtype=torch.float32),
            # PA-RL relabel pass writes optimized actions back into a
            # per-frame cache indexed by global anchor frame, so we
            # surface the anchor here (anchor_idx == -1 = unknown,
            # treated as "do not write" by the relabel pass).
            'anchor_idx': torch.tensor(int(anchor_idx), dtype=torch.long),
        }

        # PA-RL: when an optimized-action cache is loaded, emit per-
        # sample optimized chunks alongside the dataset action. The
        # downstream policy decides whether to swap the DDPM target
        # to ``action_optimized`` (controlled by its own knob).
        # Frames flagged as not-cached (valid=False) emit zeros + a
        # has_=0 gate so the loss falls back to dataset action without
        # ever introducing NaNs into the batch.
        if self.action_optimized is not None:
            ao = self.action_optimized
            av = self.action_optimized_valid
            if 0 <= anchor_idx < ao.shape[0] and bool(av[anchor_idx]):
                # mmap row -> own copy (so DataLoader workers can be
                # forked safely and so tensor-from-numpy doesn't pin
                # the mmap page). Cheap: ~640 B per sample at H=16/A=10.
                a_opt = np.asarray(ao[anchor_idx], dtype=np.float32).copy()
                torch_data['action_optimized'] = torch.from_numpy(a_opt)
                torch_data['has_action_optimized'] = torch.tensor(
                    1.0, dtype=torch.float32)
            else:
                torch_data['action_optimized'] = torch.zeros(
                    self.action_horizon, self.action_dim,
                    dtype=torch.float32,
                )
                torch_data['has_action_optimized'] = torch.tensor(
                    0.0, dtype=torch.float32)

        return torch_data
