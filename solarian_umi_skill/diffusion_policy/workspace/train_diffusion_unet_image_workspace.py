# Adapted from Universal Manipulation Interface.
# Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab.
# https://github.com/real-stanford/universal_manipulation_interface
# Licensed under the MIT License. The full text is LICENSE.umi
# beside this package.

# Stringified annotations: lets us keep type hints like
# `self.model: DiffusionUnetImagePolicy` without paying the import cost of
# `DiffusionUnetImagePolicy` (and friends) at module load. They'll be
# imported lazily inside `run()` where the runtime values actually matter.
# Required for the lazy-import block below to be safe across the file.
from __future__ import annotations

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
import copy
import random
import numpy as np
from solarian_umi_skill.diffusion_policy.workspace.base_workspace import BaseWorkspace

# NOTE: the following imports were previously at module top-level. They
# are required only by `run()` (training), not by `__init__()` or the
# inference-time accessors. Importing them at module top adds several
# seconds to cold-start of the policy server (notably `wandb` and the
# transitive `accelerate` / dataset / env_runner stack), so we defer
# them. The `if False:` block is kept to make the migration discoverable
# to static analyzers / IDEs: it documents what used to be here but is
# never executed at runtime.
if False:  # pragma: no cover - tooling hint only
    from torch.utils.data import DataLoader  # noqa: F401
    import wandb  # noqa: F401
    import pickle  # noqa: F401
    import tqdm  # noqa: F401
    import shutil  # noqa: F401
    from solarian_umi_skill.diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy  # noqa: F401
    from solarian_umi_skill.diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseDataset  # noqa: F401
    from solarian_umi_skill.diffusion_policy.env_runner.base_image_runner import BaseImageRunner  # noqa: F401
    from solarian_umi_skill.diffusion_policy.common.checkpoint_util import TopKCheckpointManager  # noqa: F401
    from solarian_umi_skill.diffusion_policy.common.json_logger import JsonLogger  # noqa: F401
    from solarian_umi_skill.diffusion_policy.common.pytorch_util import dict_apply, optimizer_to  # noqa: F401
    from solarian_umi_skill.diffusion_policy.model.diffusion.ema_model import EMAModel  # noqa: F401
    from solarian_umi_skill.diffusion_policy.model.common.lr_scheduler import get_scheduler  # noqa: F401
    from accelerate import Accelerator  # noqa: F401

OmegaConf.register_new_resolver("eval", eval, replace=True)


@torch.no_grad()
def _awr_metrics(policy) -> dict:
    """Pass-through of the policy's per-batch AWR diagnostics.

    ``DiffusionUnetTimmPolicy.compute_loss`` populates
    ``last_awr_info`` after every forward pass when AWR is active. We
    just forward those scalars to wandb, prefixed under ``awr/``. When
    AWR is off this returns an empty dict (so the metrics never appear
    on the BC training run's wandb panel and don't pollute resumes).

    Filters to ``awr/*`` keys only so PA-RL diagnostics
    (``parl/*``) don't get double-emitted by both this and
    ``_parl_metrics`` -- they share ``last_awr_info`` since both
    write into it from compute_loss.
    """
    info = getattr(policy, 'last_awr_info', None)
    if not info:
        return {}
    return {k: v for k, v in info.items() if k.startswith('awr/')}


@torch.no_grad()
def _parl_metrics(policy) -> dict:
    """Pass-through of the policy's per-batch PA-RL diagnostics.

    ``DiffusionUnetTimmPolicy.compute_loss`` populates
    ``last_awr_info`` with ``parl/*`` keys when the batch carries
    optimized actions; this just forwards them to wandb. Empty dict
    when PA-RL is off so the metrics never appear on AWR / BC runs.
    """
    info = getattr(policy, 'last_awr_info', None)
    if not info:
        return {}
    return {k: v for k, v in info.items() if k.startswith('parl/')}


@torch.no_grad()
def _bbox_mask_weight_metrics(policy) -> dict:
    """Snapshot of patch_embed.proj weight norms for bbox-mask-channel rgb keys.

    Returns one block of metrics per rgb key that has a bbox mask routed to
    it (see TimmObsEncoder.bbox_mask_map). For each key we log:

      * ``bbox_mask/<rgb_key>/w_norm_mask_ch``: L2 norm of the 4th input-
        channel slice of patch_embed.proj. Exactly 0 at training step 0
        (zero-init); growing > 0 proves the optimizer is using the mask.
      * ``bbox_mask/<rgb_key>/w_norm_rgb_mean``: mean L2 norm of the three
        RGB input-channel slices. Stable reference scale to compare against.
      * ``bbox_mask/<rgb_key>/w_ratio``: ratio of the two (mask norm /
        mean rgb norm). A "healthy" trained value is roughly 0.05 - 1.0;
        values close to 0 mean the optimizer declined to use the mask
        (almost certainly because the training data didn't reward it).

    Returns an empty dict when no bbox is routed via the mask channel
    (e.g. ``bbox_mode: low_dim`` runs), so it's safe to call unconditionally.
    """
    encoder = getattr(policy, 'obs_encoder', None)
    if encoder is None:
        return {}
    mask_map = getattr(encoder, 'bbox_mask_map', None)
    if not mask_map:
        return {}

    metrics = {}
    for rgb_key in mask_map.keys():
        # nn.ModuleDict doesn't expose .get(); use __contains__ + []
        if rgb_key not in encoder.key_model_map:
            continue
        model = encoder.key_model_map[rgb_key]
        if not hasattr(model, 'patch_embed') \
                or not hasattr(model.patch_embed, 'proj'):
            continue
        weight = model.patch_embed.proj.weight  # (out_ch, in_ch=4, k, k)
        if weight.shape[1] < 4:
            continue
        rgb_norms = [weight[:, c].norm().item() for c in range(3)]
        mask_norm = weight[:, 3].norm().item()
        rgb_mean = sum(rgb_norms) / 3.0
        ratio = (mask_norm / rgb_mean) if rgb_mean > 0 else 0.0
        prefix = f'bbox_mask/{rgb_key}'
        metrics[f'{prefix}/w_norm_mask_ch'] = mask_norm
        metrics[f'{prefix}/w_norm_rgb_mean'] = rgb_mean
        metrics[f'{prefix}/w_ratio'] = ratio
    return metrics


class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        # self.optimizer = hydra.utils.instantiate(
        #     cfg.optimizer, params=self.model.parameters())

        obs_encorder_lr = cfg.optimizer.lr
        if cfg.policy.obs_encoder.pretrained:
            obs_encorder_lr *= 0.1
            print('==> reduce pretrained obs_encorder\'s lr')
        obs_encorder_params = list()
        for param in self.model.obs_encoder.parameters():
            if param.requires_grad:
                obs_encorder_params.append(param)
        print(f'obs_encorder params: {len(obs_encorder_params)}')
        param_groups = [
            {'params': self.model.model.parameters()},
            {'params': obs_encorder_params, 'lr': obs_encorder_lr}
        ]
        # self.optimizer = hydra.utils.instantiate(
        #     cfg.optimizer, params=param_groups)
        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop('_target_')
        self.optimizer = torch.optim.AdamW(
            params=param_groups,
            **optimizer_cfg
        )

        # configure training state
        self.global_step = 0
        self.epoch = 0

        # PA-RL: frozen IQL critic is materialized lazily by run() once
        # the device is settled. Kept as None on workspaces that don't
        # opt in, so legacy AWR / BC runs pay nothing.
        self.q_critic = None

        # PA-RL: optional one-time cache of ``critic.state_embed`` for
        # every dataset frame, built once after the (frozen) critic is
        # loaded and reused across all relabel cycles. Stored as a
        # CPU tensor of shape ``(N_frames, D_state)`` so per-batch
        # lookup is a cheap index_select; ``valid`` flags rows whose
        # anchor_idx couldn't be resolved during the build pass.
        # ``None`` on workspaces that don't opt in (``parl.cache_state_embed=false``).
        # Both fields are ``Optional[torch.Tensor]`` once populated by run().
        self._parl_state_embed_cache = None
        self._parl_state_embed_valid = None

        # PA-RL: optional one-time cache of the proposer's
        # ``obs_encoder(nobs)`` output for every dataset frame. Built
        # ONCE after the (frozen) proposer encoder is loaded and reused
        # across ALL relabel cycles. Eliminates the second ViT-L
        # forward in propose_actions (the proposer's encoder is the
        # SAME size as the critic's), so combined with the state_embed
        # cache the relabel pass does ZERO ViT-L forwards per iter --
        # all encoder work is amortized into the one-time precompute.
        # Only valid when ``training.freeze_encoder=true``;
        # automatically disabled otherwise.
        self._parl_global_cond_cache = None
        self._parl_global_cond_valid = None

        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']

    def _precompute_obs_feature_cache(
        self,
        *,
        train_dataset,
        val_dataset,
        forward_fn,
        cache_label: str,
        n_frames: int,
        device,
        accelerator,
        parl_cfg,
    ):
        """Generic helper: scan train + val once and cache an
        ``(N_frames, D)`` tensor produced by a frozen encoder via
        ``forward_fn(obs_dict_on_device) -> (B, D)``.

        Both ``_precompute_state_embed_cache`` and
        ``_precompute_global_cond_cache`` use this skeleton; only the
        forward function differs.

        Notes
        -----
        - Augmentation realism: the dataset's ``__getitem__`` injects
          random start-pose noise (``umi_dataset.py`` ~L538) and the
          obs encoder may apply RandomCrop / ColorJitter. The cache
          therefore stores ONE realization of those random augs per
          frame. This matches the noise budget the encoder's owner
          (IQL critic / BC policy) was trained against (each train
          step also sees a single random aug per frame), so a single
          realization is statistically equivalent to a fresh draw.
        - Validity: any sample whose ``anchor_idx`` is ``-1`` (sampler
          sentinel) leaves its cache row uninitialized; ``valid``
          flags it false and the relabel pass falls back to
          recomputing on the fly for that batch.

        Args:
            forward_fn: callable taking an ``obs_dict`` of tensors
                on ``device`` and returning a single ``(B, D)``
                tensor on the same device. Wrapped externally in
                ``torch.no_grad`` + ``accelerator.autocast``; do not
                wrap inside the callable.
            cache_label: short human label used in log lines and the
                tqdm description (e.g. ``"state_embed"`` or
                ``"global_cond"``).

        Returns:
            ``(cache, valid)`` -- pinned CPU ``(N_frames, D)`` tensor
            and ``(N_frames,)`` bool mask.
        """
        from torch.utils.data import DataLoader
        import time as _time
        import tqdm as _tqdm
        from solarian_umi_skill.diffusion_policy.common.pytorch_util import dict_apply

        precompute_bs = int(parl_cfg.get(
            'precompute_batch_size',
            parl_cfg.get('relabel_batch_size', 16),
        ))
        precompute_workers = int(parl_cfg.get(
            'precompute_num_workers',
            parl_cfg.get('relabel_num_workers', 4),
        ))

        # Discover the output dim by running ONE tiny batch. Avoids
        # hardcoding the architecture's hidden dim in the workspace
        # and keeps this method drop-in for any future encoder.
        probe = train_dataset[0]
        probe_obs = dict_apply(
            probe['obs'],
            lambda x: x.unsqueeze(0).to(device, non_blocking=True),
        )
        with torch.no_grad(), accelerator.autocast():
            probe_emb = forward_fn(probe_obs)
        D = int(probe_emb.shape[-1])
        cache_dtype = probe_emb.dtype  # bf16 under autocast on A100
        del probe, probe_obs, probe_emb

        # GPU-resident cache. We considered pinned CPU tensors here
        # but ran into NaN values reading back from pinned bf16
        # tensors on this stack -- pytorch's pin_memory has had
        # subtle bf16 issues across versions. GPU storage fits
        # comfortably (a 200k-frame, D=1024 bf16 cache is ~410 MB)
        # AND eliminates the per-batch H2D copy, so this is the
        # cleaner default. The relabel pass is single-rank in v1,
        # so only the main-process GPU holds the cache.
        cache = torch.zeros(
            (n_frames, D), dtype=cache_dtype, device=device)
        valid = torch.zeros(
            (n_frames,), dtype=torch.bool, device=device)

        splits = []
        if len(train_dataset) > 0:
            splits.append(('train', train_dataset))
        if val_dataset is not None and len(val_dataset) > 0:
            splits.append(('val', val_dataset))

        n_seen = 0
        t0_total = _time.perf_counter()
        for split_name, ds in splits:
            # Drop any in-memory action_optimized so the dataset
            # serves the BASE obs (action_optimized doesn't actually
            # change obs but we mirror the relabel pass's defensive
            # save/restore for symmetry).
            saved_actions = ds.action_optimized
            saved_valid = ds.action_optimized_valid
            ds.action_optimized = None
            ds.action_optimized_valid = None
            try:
                loader = DataLoader(
                    ds,
                    batch_size=precompute_bs,
                    shuffle=False,
                    num_workers=precompute_workers,
                    pin_memory=True,
                    persistent_workers=False,
                    drop_last=False,
                )
                accelerator.print(
                    f"[PA-RL] precompute {cache_label} cache for {split_name}: "
                    f"{len(ds)} samples / {len(loader)} batches "
                    f"(D={D}, dtype={cache_dtype})"
                )
                t0 = _time.perf_counter()
                pbar = _tqdm.tqdm(
                    loader,
                    total=len(loader),
                    desc=f"PA-RL {cache_label} {split_name}",
                    dynamic_ncols=True,
                    mininterval=1.0,
                    leave=False,
                    disable=not accelerator.is_main_process,
                )
                for batch in pbar:
                    obs = dict_apply(
                        batch['obs'],
                        lambda x: x.to(device, non_blocking=True),
                    )
                    anchor_idx_t = batch['anchor_idx'].to(
                        device, non_blocking=True).long()
                    with torch.no_grad(), accelerator.autocast():
                        emb = forward_fn(obs)
                    # Vectorized scatter on GPU: ``cache`` lives on
                    # ``device`` so we never round-trip through CPU.
                    # ``in_range_t`` filters out the sentinel
                    # anchor_idx == -1 rows (sampler emits this when
                    # the global frame can't be resolved); those rows
                    # stay zero in the cache and ``valid`` flags them
                    # false so the relabel pass falls back to fresh
                    # compute for that batch.
                    in_range_t = (anchor_idx_t >= 0) & (
                        anchor_idx_t < n_frames)
                    if bool(in_range_t.any().item()):
                        valid_anchors_t = anchor_idx_t[in_range_t]
                        cache[valid_anchors_t] = emb.detach()[in_range_t]
                        valid[valid_anchors_t] = True
                    n_seen += int(emb.shape[0])
                pbar.close()
                dt = _time.perf_counter() - t0
                accelerator.print(
                    f"[PA-RL] {cache_label} {split_name} done in {dt:.1f}s "
                    f"({len(ds) / max(dt, 1e-3):.1f} samples/s)"
                )
            finally:
                ds.action_optimized = saved_actions
                ds.action_optimized_valid = saved_valid

        n_valid = int(valid.sum().item())
        total_dt = _time.perf_counter() - t0_total
        # Sanity check: any non-finite values in the cache silently
        # propagate to NaN Q-values during relabel and look like the
        # critic has gone insane. Cheap to check once at build time.
        # Only inspect the valid rows (zeros in unused rows are fine).
        if n_valid > 0:
            valid_rows = cache[valid]
            n_bad = int((~torch.isfinite(valid_rows)).any(dim=-1).sum().item())
            if n_bad > 0:
                # Hard-fail rather than silently produce NaN Q-values:
                # there is no recovery from a corrupt feature cache.
                raise RuntimeError(
                    f"[PA-RL] {cache_label} cache has {n_bad}/{n_valid} "
                    f"valid rows with non-finite values (NaN/Inf). "
                    f"This usually means the encoder ran into bf16 "
                    f"overflow on this stack. Try setting "
                    f"parl.cache_{cache_label}=false to disable the "
                    f"cache, or upgrade pytorch."
                )
        accelerator.print(
            f"[PA-RL] {cache_label} cache built in {total_dt:.1f}s: "
            f"{n_valid}/{n_frames} frames valid "
            f"({100.0 * n_valid / max(n_frames, 1):.1f}%), "
            f"{cache.element_size() * cache.nelement() / 1e6:.1f} MB on "
            f"{cache.device}"
        )
        return cache, valid

    def _precompute_state_embed_cache(
        self,
        *,
        train_dataset,
        val_dataset,
        critic,
        n_frames: int,
        device,
        accelerator,
        parl_cfg,
    ):
        """Precompute and cache ``critic.state_embed`` for every dataset frame.

        The IQL critic's obs encoder is FROZEN throughout PA-RL fine-
        tuning (``parl.freeze_critic=true``), so its state embedding
        for any given frame is constant across the entire run. Each
        relabel cycle replaces the per-batch
        ``critic.state_embed(obs_dict)`` call (a heavy ViT-L forward)
        with an O(B) ``index_select``.
        """
        critic.eval()
        return self._precompute_obs_feature_cache(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            forward_fn=lambda obs: critic.state_embed(obs, target=False),
            cache_label="state_embed",
            n_frames=n_frames,
            device=device,
            accelerator=accelerator,
            parl_cfg=parl_cfg,
        )

    def _precompute_global_cond_cache(
        self,
        *,
        train_dataset,
        val_dataset,
        proposer,
        n_frames: int,
        device,
        accelerator,
        parl_cfg,
    ):
        """Precompute and cache the proposer's ``obs_encoder(nobs)`` output.

        When ``training.freeze_encoder=True`` the policy's (and EMA
        proposer's) ViT-L visual encoder is frozen for the entire
        PA-RL run. Its output for any given frame is therefore
        constant, so we amortize one full pass over the dataset and
        afterwards every relabel cycle skips the heavy
        ``proposer.obs_encoder(nobs)`` forward inside ``propose_actions``,
        replacing it with an O(B) ``index_select``. This is the same
        optimization as ``_precompute_state_embed_cache`` but for the
        SECOND big ViT-L the relabel pass would otherwise run.

        The dim D_cond depends on the policy's
        ``timm_obs_encoder.output_shape()`` and is discovered at
        runtime by probing the first sample, so the helper works
        unchanged for any future encoder shape.
        """
        proposer.eval()

        def _forward(obs):
            nobs = proposer.normalizer.normalize(obs)
            return proposer.obs_encoder(nobs)

        return self._precompute_obs_feature_cache(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            forward_fn=_forward,
            cache_label="global_cond",
            n_frames=n_frames,
            device=device,
            accelerator=accelerator,
            parl_cfg=parl_cfg,
        )

    def _parl_relabel(
        self,
        *,
        train_dataset,
        val_dataset,
        proposer,
        critic,
        n_frames: int,
        action_horizon: int,
        action_dim: int,
        device,
        accelerator,
        parl_cfg,
        state_embed_cache=None,
        state_embed_valid=None,
        global_cond_cache=None,
        global_cond_valid=None,
    ):
        """Run ONE PA-RL relabel pass and return ``(actions, valid, info)``.

        Iterates the train + val datasets sequentially (no shuffle, no
        DDP sharding -- single-rank pass on accelerator's main
        process), proposes K candidates per state from ``proposer``,
        runs PA-RL action optimization against the frozen ``critic``,
        and scatters the chosen actions into a per-frame
        ``(n_frames, H, A)`` array indexed by the global anchor
        ``anchor_idx`` emitted by ``UmiDataset.__getitem__``.

        Algorithm 1 + Eq. 4.3 from Sobol Mark et al. 2024.
        Implementation lives in
        ``diffusion_policy.policy.parl_action_optimizer``.

        Returns:
            actions: (n_frames, H, A) float32; rows where valid=False
                are zeros (defensive default; the dataset emits
                ``has_action_optimized=0`` for those, so the loss
                falls back to dataset action).
            valid: (n_frames,) bool.
            info: dict of aggregated diagnostics under the ``parl/``
                prefix, suitable for ``accelerator.log``.
        """
        from torch.utils.data import DataLoader
        import time as _time
        import tqdm as _tqdm
        from solarian_umi_skill.diffusion_policy.policy.parl_action_optimizer import optimize_actions
        from solarian_umi_skill.diffusion_policy.common.pytorch_util import dict_apply

        K = int(parl_cfg.get('k_samples', 32))
        m_top = int(parl_cfg.get('m_top', 10))
        T = int(parl_cfg.get('T_grad_steps', 10))
        alpha = float(parl_cfg.get('alpha_grad', 3.0e-4))
        pick_mode = str(parl_cfg.get('distill_mode', 'argmax'))
        softmax_temp = float(parl_cfg.get('softmax_temperature', 1.0))
        include_data_action = bool(parl_cfg.get('include_dataset_action', True))
        proposer_steps = parl_cfg.get('proposer_inference_steps', None)
        if proposer_steps is not None:
            proposer_steps = int(proposer_steps)
        relabel_batch_size = int(parl_cfg.get('relabel_batch_size', 16))
        relabel_workers = int(parl_cfg.get('relabel_num_workers', 4))

        # Make sure the proposer / critic are in eval mode and on the
        # right device. We never set requires_grad on either (proposer
        # has frozen visual when freeze_encoder=True; critic is
        # explicitly frozen at load time when freeze_critic=True).
        proposer.eval()
        critic.eval()

        out_actions = np.zeros((n_frames, action_horizon, action_dim),
                               dtype=np.float32)
        out_valid = np.zeros(n_frames, dtype=bool)

        # Single aggregator across train + val for wandb (we don't
        # need to break out per-split numbers in the diagnostics; the
        # split breakdown is in the per-pass log line below).
        agg = {}

        splits = []
        if len(train_dataset) > 0:
            splits.append(('train', train_dataset))
        if val_dataset is not None and len(val_dataset) > 0:
            splits.append(('val', val_dataset))

        n_seen = 0
        t0_total = _time.perf_counter()
        for split_name, ds in splits:
            # Disable any cache that the dataset already has so this
            # pass sees the BASE dataset action (not yesterday's
            # optimized chunk masquerading as the dataset). Restored
            # by the caller after the pass when it calls
            # ``set_action_optimized`` with the new array.
            saved_actions = ds.action_optimized
            saved_valid = ds.action_optimized_valid
            ds.action_optimized = None
            ds.action_optimized_valid = None
            try:
                loader = DataLoader(
                    ds,
                    batch_size=relabel_batch_size,
                    shuffle=False,
                    num_workers=relabel_workers,
                    pin_memory=True,
                    persistent_workers=False,
                    drop_last=False,
                )
                accelerator.print(
                    f"[PA-RL] relabel {split_name}: "
                    f"{len(ds)} samples / {len(loader)} batches "
                    f"(K={K}, m={m_top}, T={T}, alpha={alpha:g}, "
                    f"DDIM_steps={proposer_steps if proposer_steps is not None else 'policy_default'})"
                )
                t0 = _time.perf_counter()
                # Live tqdm bar over the relabel batches. Postfix shows
                # the running mean Q before / after local optimization
                # so the operator can confirm Q is improving (not just
                # that the loop is moving) without needing to wait for
                # the per-cycle wandb log at the end.
                pbar = _tqdm.tqdm(
                    loader,
                    total=len(loader),
                    desc=f"PA-RL relabel {split_name}",
                    dynamic_ncols=True,
                    mininterval=1.0,
                    leave=False,
                    disable=not accelerator.is_main_process,
                )
                # Running Q sums for the postfix; intentionally weighted
                # by batch size so partial batches don't bias the mean.
                q_before_running = 0.0
                q_after_running = 0.0
                q_chosen_running = 0.0
                q_n_running = 0
                for batch_idx, batch in enumerate(pbar):
                    obs = dict_apply(
                        batch['obs'],
                        lambda x: x.to(device, non_blocking=True),
                    )
                    anchor_idx_t = batch['anchor_idx']
                    anchor_idx = anchor_idx_t.cpu().numpy()
                    dataset_action = None
                    if include_data_action:
                        dataset_action = batch['action'].to(
                            device, non_blocking=True)

                    # Cache lookup: when the frozen-critic state_embed
                    # cache is populated AND every row in this batch
                    # has a valid cached embedding, we hand it to
                    # optimize_actions and skip the heavy ViT-L
                    # forward inside ``critic.state_embed``. If ANY
                    # row is invalid (anchor_idx == -1, or never
                    # written), we fall back to recomputing on the
                    # fly so correctness is preserved at the cost of
                    # one extra forward for that batch.
                    # Cache lookups for the two big ViT-L outputs:
                    #   - state_embed_cache  -> critic.state_embed(obs)
                    #   - global_cond_cache  -> proposer.obs_encoder(nobs)
                    # Both are computed once at run() startup and
                    # constant for the rest of training (their owners
                    # are frozen). Caches live ON the GPU so per-batch
                    # we just do an O(B) GPU-side ``index_select``;
                    # no H2D copy. Falls back to recomputing on the
                    # fly if any row in this batch has an unresolved
                    # ``anchor_idx``.
                    ai_long = anchor_idx_t.to(
                        device, non_blocking=True).long()
                    state_embed_override = None
                    if state_embed_cache is not None:
                        in_range = (ai_long >= 0) & (
                            ai_long < state_embed_cache.shape[0])
                        if bool(in_range.all().item()):
                            sv = state_embed_valid.index_select(0, ai_long)
                            if bool(sv.all().item()):
                                state_embed_override = (
                                    state_embed_cache.index_select(
                                        0, ai_long))
                    global_cond_override = None
                    if global_cond_cache is not None:
                        in_range_g = (ai_long >= 0) & (
                            ai_long < global_cond_cache.shape[0])
                        if bool(in_range_g.all().item()):
                            gv = global_cond_valid.index_select(0, ai_long)
                            if bool(gv.all().item()):
                                global_cond_override = (
                                    global_cond_cache.index_select(
                                        0, ai_long))

                    # Honor the accelerator's mixed_precision setting
                    # for the relabel pass too (training already does
                    # via accelerator.prepare). The proposer's ViT-L
                    # and the critic's state_embed dominate runtime
                    # and benefit ~3-4x from bf16 on A100. The local
                    # gradient ascent on the action runs through the
                    # critic's q_heads inside the autocast region;
                    # autograd handles dtype promotion to fp32 for
                    # the leaf action grad correctly.
                    with accelerator.autocast():
                        action_opt, info = optimize_actions(
                            proposer=proposer,
                            critic=critic,
                            obs_dict=obs,
                            K=K, m=m_top, T=T, alpha=alpha,
                            pick_mode=pick_mode,
                            softmax_temperature=softmax_temp,
                            dataset_action=dataset_action,
                            proposer_inference_steps=proposer_steps,
                            state_embed_override=state_embed_override,
                            global_cond_override=global_cond_override,
                        )
                    a_np = action_opt.detach().float().cpu().numpy()
                    # Vectorized scatter: O(B) Python loops add tens
                    # of microseconds per row that don't scale with
                    # GPU work, so they punish us at large B. Build a
                    # boolean mask once and let numpy do the dispatch.
                    in_range = (anchor_idx >= 0) & (anchor_idx < n_frames)
                    if in_range.any():
                        valid_anchors = anchor_idx[in_range]
                        out_actions[valid_anchors] = a_np[in_range]
                        out_valid[valid_anchors] = True
                    n_seen += int(a_np.shape[0])
                    for k_, v_ in info.items():
                        if isinstance(v_, (bool,)):
                            v_ = float(v_)
                        if isinstance(v_, (int, float)):
                            agg.setdefault(k_, []).append(float(v_))

                    # Update tqdm postfix every batch from the same
                    # ``info`` dict optimize_actions just returned. The
                    # running averages here weight each batch by its
                    # size so the displayed numbers are unbiased even
                    # if the last batch is partial.
                    bsz = int(a_np.shape[0])
                    q_before_running += float(
                        info.get('parl/q_before_local_mean', 0.0)) * bsz
                    q_after_running += float(
                        info.get('parl/q_after_local_mean', 0.0)) * bsz
                    q_chosen_running += float(
                        info.get('parl/q_chosen_mean', 0.0)) * bsz
                    q_n_running += bsz
                    if q_n_running > 0:
                        pbar.set_postfix({
                            'Q_before': f"{q_before_running / q_n_running:+.3f}",
                            'Q_after': f"{q_after_running / q_n_running:+.3f}",
                            'Q_chosen': f"{q_chosen_running / q_n_running:+.3f}",
                            'rev': f"{float(info.get('parl/frac_local_reverted', 0.0)):.2f}",
                            'data_picked': f"{float(info.get('parl/frac_dataset_action_chosen', 0.0)):.2f}",
                        }, refresh=False)
                pbar.close()
                dt = _time.perf_counter() - t0
                accelerator.print(
                    f"[PA-RL] relabel {split_name} done in {dt:.1f}s "
                    f"({len(ds) / max(dt, 1e-3):.1f} samples/s)"
                )
            finally:
                ds.action_optimized = saved_actions
                ds.action_optimized_valid = saved_valid

        info_means = {k: float(np.mean(v)) for k, v in agg.items() if v}
        info_means['parl/n_relabeled_frames'] = int(out_valid.sum())
        info_means['parl/n_total_frames'] = int(n_frames)
        info_means['parl/n_samples_seen'] = int(n_seen)
        info_means['parl/relabel_seconds'] = float(
            _time.perf_counter() - t0_total)
        return out_actions, out_valid, info_means

    def run(self):
        # Lazy imports: these are training-only dependencies. Keeping them
        # out of module-top imports saves several seconds of cold start when
        # the workspace class is loaded purely for inference (e.g. by
        # policy_server.py -> hydra.utils.get_class). The first call to
        # run() pays the import cost; the imports are then cached in
        # sys.modules so subsequent calls are free.
        from torch.utils.data import DataLoader
        from solarian.tracking.accelerate import accelerate_trackers
        import pickle
        import tqdm
        import shutil
        from solarian_umi_skill.diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
        from solarian_umi_skill.diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseDataset
        from solarian_umi_skill.diffusion_policy.dataset.umi_dataset import mix_sampler
        from solarian_umi_skill.diffusion_policy.env_runner.base_image_runner import BaseImageRunner
        from solarian_umi_skill.diffusion_policy.common.checkpoint_util import TopKCheckpointManager
        from solarian_umi_skill.diffusion_policy.common.json_logger import JsonLogger
        from solarian_umi_skill.diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
        from solarian_umi_skill.diffusion_policy.model.diffusion.ema_model import EMAModel
        from solarian_umi_skill.diffusion_policy.model.common.lr_scheduler import get_scheduler
        from solarian_umi_skill.diffusion_policy.model.vision.timm_obs_encoder import widen_cond_for_box_pool
        from accelerate import Accelerator, DistributedDataParallelKwargs
        import dill

        cfg = copy.deepcopy(self.cfg)

        # When the visual encoder is frozen (typical for AWR finetuning),
        # its parameters don't receive gradients. The default DDP wrap
        # then fails with "Expected to have finished reduction in the
        # prior iteration..." because DDP's reducer expects every
        # registered parameter to participate in backward. The fix is
        # to construct DDP with find_unused_parameters=True so it
        # tolerates the no-grad params; we gate this on freeze_encoder
        # to avoid the (small) per-step graph-traversal overhead in
        # plain BC training where everything trains.
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=bool(
                cfg.training.get('freeze_encoder', False)
            )
        )
        # logging.backend (wandb, solarian_base, none, or a list) picks where
        # the run goes; tracking drops empty tags and clamps long ones.
        logging_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        log_with, init_kwargs = accelerate_trackers(logging_cfg)
        accelerator = Accelerator(
            log_with=log_with,
            kwargs_handlers=[ddp_kwargs],
        )
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs=init_kwargs,
        )

        # resume training
        if cfg.training.resume:
            # Support custom checkpoint path via config, or use latest from output_dir
            if hasattr(cfg.training, 'checkpoint_path') and cfg.training.checkpoint_path:
                ckpt_path = pathlib.Path(cfg.training.checkpoint_path)
                if ckpt_path.is_file():
                    accelerator.print(f"Resuming from checkpoint {ckpt_path}")
                    self.load_checkpoint(path=ckpt_path)
                else:
                    accelerator.print(f"Warning: Checkpoint path {ckpt_path} not found, skipping resume")
            else:
                lastest_ckpt_path = self.get_checkpoint_path()
                if lastest_ckpt_path.is_file():
                    accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
                    self.load_checkpoint(path=lastest_ckpt_path)
            # Checkpoints are saved after an epoch's training steps but before
            # its end-of-epoch epoch / global_step increments.
            if self.epoch > 0 or self.global_step > 0:
                self.epoch += 1
                self.global_step += 1
                accelerator.print(f"Continuing at epoch {self.epoch}, step {self.global_step}")
        elif (
            hasattr(cfg.training, 'init_from_ckpt')
            and cfg.training.init_from_ckpt
        ):
            # Fresh run that *initializes* model weights from another
            # checkpoint without resuming optimizer / scheduler state.
            # Used for AWR finetuning of a pre-trained BC policy:
            # we want the BC weights as a starting point but a fresh
            # optimizer (so the LR schedule restarts cleanly), fresh
            # global_step / epoch counters, and (importantly) the *new*
            # cfg's policy hyperparameters in self.model -- including
            # awr_beta etc.
            init_path = pathlib.Path(cfg.training.init_from_ckpt)
            if init_path.is_file():
                accelerator.print(
                    f"Initializing model weights from {init_path} "
                    f"(optimizer + step counters NOT loaded)"
                )
                # Skip optimizer state -- the new run uses a fresh
                # optimizer with possibly-different LR. include_keys=()
                # skips global_step / epoch pickles too. The AWR
                # config knobs are plain Python attributes (set in
                # __init__ from the new cfg, not stored in state_dict),
                # so a strict load of the BC ckpt's `model` and
                # `ema_model` state_dicts produces no key mismatch.
                payload = torch.load(init_path.open('rb'), pickle_module=dill)
                if self.model.obs_encoder.feature_aggregation == 'box_pool':
                    target = self.model.state_dict()
                    for key in ('model', 'ema_model'):
                        if key in payload['state_dicts']:
                            widened = widen_cond_for_box_pool(
                                payload['state_dicts'][key], self.model.obs_encoder, target)
                            accelerator.print(
                                f"box_pool: widened {widened} {key} conditioning weights "
                                f"of {init_path.name}")
                self.load_payload(payload, exclude_keys=['optimizer'], include_keys=())
                # Force a clean restart of the schedule even though
                # global_step/epoch were already 0 from __init__.
                self.global_step = 0
                self.epoch = 0

                # BC-anchored AWR: snapshot the freshly-loaded BC
                # weights into a frozen reference U-Net on the policy.
                # No-op when ``policy.awr_anchor_lambda <= 0``, so
                # legacy AWR / BC runs pay nothing extra. We do this
                # here (not in policy.__init__) because the policy is
                # built from the new AWR cfg with random weights; the
                # anchor must be the BC ckpt's weights, not the random
                # init.
                init_fn = getattr(self.model, 'init_frozen_ref', None)
                if callable(init_fn):
                    accelerator.print(
                        "Calling policy.init_frozen_ref() to snapshot "
                        "BC weights for anchor loss (no-op when "
                        "awr_anchor_lambda <= 0).")
                    init_fn()
                    # Mirror onto the EMA shadow so its forward pass
                    # under sample_every / validation can also access
                    # the anchor reference. The deep-copied ref itself
                    # is shared by attribute reference -- it's frozen
                    # and never updated, so sharing is safe.
                    if (
                        cfg.training.use_ema
                        and getattr(self, 'ema_model', None) is not None
                    ):
                        object.__setattr__(
                            self.ema_model, 'frozen_ref_model',
                            getattr(self.model, 'frozen_ref_model', None),
                        )
            else:
                accelerator.print(
                    f"Warning: init_from_ckpt={init_path} not found; "
                    f"training will start from random / pretrained weights."
                )

        # ============== PA-RL setup ==============
        # Resolve PA-RL config block (may be absent). Fields:
        #   parl.enabled              bool   master switch
        #   parl.relabel_every        int    epochs between relabels
        #   parl.k_samples / m_top / T_grad_steps / alpha_grad
        #   parl.distill_mode         'argmax' | 'softmax'
        #   parl.softmax_temperature  float
        #   parl.proposer             'ema' | 'main'
        #   parl.include_dataset_action bool
        #   parl.proposer_inference_steps  int (override DDIM steps)
        #   parl.relabel_batch_size / num_workers  int
        #   parl.freeze_critic        bool   (always True for v1)
        # Q-critic checkpoint path lives at ``training.q_critic_ckpt``
        # for symmetry with ``training.init_from_ckpt`` (BC ckpt).
        parl_cfg_node = None
        try:
            if OmegaConf.select(cfg, 'parl') is not None:
                parl_cfg_node = cfg.parl
        except Exception:
            parl_cfg_node = None
        parl_enabled = bool(
            parl_cfg_node is not None
            and OmegaConf.select(parl_cfg_node, 'enabled', default=False)
        )
        q_ckpt_path = OmegaConf.select(
            cfg, 'training.q_critic_ckpt', default=None)

        if parl_enabled and q_ckpt_path:
            from solarian_umi_skill.diffusion_policy.common.q_critic_util import load_iql_critic
            init_path = pathlib.Path(os.path.expanduser(str(q_ckpt_path)))
            if not init_path.is_file():
                raise FileNotFoundError(
                    f"PA-RL: training.q_critic_ckpt={init_path} not found."
                )
            # Load critic to CPU first; we'll move it to device once
            # `accelerator.prepare()` has resolved the actual device
            # below. This also matches policy_server.load_q_critic's
            # contract (CPU -> device after build).
            accelerator.print(
                f"[PA-RL] loading frozen Q critic from {init_path}")
            q_critic, q_cfg = load_iql_critic(
                str(init_path), torch.device('cpu'),
            )
            # Sanity-check action shape vs the diffusion policy.
            policy_h = int(self.model.action_horizon)
            policy_a = int(self.model.action_dim)
            critic_h = int(q_critic.action_horizon)
            critic_a = int(q_critic.action_dim)
            if (critic_h, critic_a) != (policy_h, policy_a):
                raise RuntimeError(
                    f"PA-RL: Q-critic action shape ({critic_h}, {critic_a}) "
                    f"!= diffusion policy ({policy_h}, {policy_a}); "
                    f"the relabel pass would produce nonsense Q values."
                )
            # Freeze critic params (default; the freeze_critic flag is
            # plumbed for forward-compat with future Cal-QL co-training).
            freeze_critic = bool(OmegaConf.select(
                parl_cfg_node, 'freeze_critic', default=True))
            if freeze_critic:
                for p in q_critic.parameters():
                    p.requires_grad_(False)
            self.q_critic = q_critic
            accelerator.print(
                f"[PA-RL] q_critic ready (frozen={freeze_critic}); "
                f"action=({critic_h}, {critic_a})"
            )
        elif parl_enabled and not q_ckpt_path:
            accelerator.print(
                "[PA-RL] parl.enabled=true but training.q_critic_ckpt is "
                "unset; PA-RL will be DISABLED (relabel pass skipped, "
                "dataset emits no action_optimized)."
            )
            parl_enabled = False

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset) or isinstance(dataset, BaseDataset)
        # task.dataset_mix: [{path, weight}], the first entry being
        # task.dataset_path. The others train on all their episodes;
        # validation comes from the first.
        dataset_mix = cfg.task.get('dataset_mix') or [
            {'path': cfg.task.dataset_path, 'weight': 1.0}]
        train_parts = [dataset] + [
            hydra.utils.instantiate(
                cfg.task.dataset, dataset_path=entry['path'],
                val_dataset_path=None, val_ratio=0.0)
            for entry in dataset_mix[1:]
        ]
        if len(train_parts) > 1 and parl_enabled:
            raise RuntimeError(
                "PA-RL caches are indexed by the frames of one dataset; "
                "it cannot run with task.dataset_mix.")
        samples_per_epoch = cfg.task.get('samples_per_epoch')
        dataloader_kwargs = dict(cfg.dataloader)
        if len(train_parts) > 1 or samples_per_epoch:
            dataloader_kwargs.pop('shuffle', None)
            dataloader_kwargs['sampler'] = mix_sampler(
                [len(part) for part in train_parts],
                [entry['weight'] for entry in dataset_mix],
                samples_per_epoch)
        train_dataloader = DataLoader(
            torch.utils.data.ConcatDataset(train_parts) if len(train_parts) > 1 else dataset,
            **dataloader_kwargs)

        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            part_data = [part.normalizer_data() for part in train_parts]
            normalizer = dataset.fit_normalizer({
                key: np.concatenate([data[key] for data in part_data])
                for key in part_data[0]})
            pickle.dump(normalizer, open(normalizer_path, 'wb'))

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        print('train dataset:', [len(part) for part in train_parts],
              'train dataloader:', len(train_dataloader))
        print('val dataset:', len(val_dataset), 'val dataloader:', len(val_dataloader))

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # Set initial_lr for each param group if resuming (required by LR scheduler)
        if cfg.training.resume and self.global_step > 0:
            for param_group in self.optimizer.param_groups:
                if 'initial_lr' not in param_group:
                    param_group['initial_lr'] = param_group['lr']

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        # configure checkpoint
        # Build constant fields used in checkpoint filenames from the experiment
        # config (task name, model type, visual encoder, total batch size,
        # dataset file name without extension). These are optional in cfg, so
        # missing values fall back to 'na' to avoid breaking other workspaces
        # that share this manager.
        dataset_path = cfg.task.get('dataset_path', '') if hasattr(cfg, 'task') else ''
        dataset_file_name = os.path.basename(str(dataset_path))
        # Strip common archive/zarr extensions (e.g. .zarr.zip, .zarr, .zip).
        while True:
            stem, ext = os.path.splitext(dataset_file_name)
            if ext.lower() in ('.zip', '.zarr', '.gz', '.tar'):
                dataset_file_name = stem
            else:
                break
        if not dataset_file_name:
            dataset_file_name = 'na'

        total_batch = int(cfg.get('batch_size', 0)) * int(cfg.get('num_gpus', 1))

        static_format_kwargs = {
            'task_name': cfg.get('task_name', 'na'),
            'model_type': cfg.get('model_type', 'na'),
            'visual_encoder': cfg.get('visual_encoder', 'na'),
            'total_batch': total_batch,
            'dataset_file_name': dataset_file_name,
        }

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            static_format_kwargs=static_format_kwargs,
            **cfg.checkpoint.topk
        )

        # device transfer
        # device = torch.device(cfg.training.device)
        # self.model.to(device)
        # if self.ema_model is not None:
        #     self.ema_model.to(device)
        # optimizer_to(self.optimizer, device)

        # accelerator
        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler
        )
        device = self.model.device
        if self.ema_model is not None:
            self.ema_model.to(device)

        # PA-RL: move the (already-frozen) Q critic to the same device.
        # Done AFTER accelerator.prepare so single- and multi-GPU runs
        # both pick up the right device id without re-reading config.
        # We never DDP-wrap the critic: it's frozen in v1, used only in
        # eval(), and the relabel pass runs single-rank on the main
        # process anyway.
        if self.q_critic is not None:
            self.q_critic.to(device)
            self.q_critic.eval()

        # PA-RL: optional one-time precompute of ``critic.state_embed``
        # for every dataset frame. Saves the heavy ViT-L state_embed
        # forward on EVERY relabel cycle thereafter; replaced by an
        # O(B) ``index_select`` from a pinned CPU tensor and a single
        # H2D copy. Only runs when ``parl.cache_state_embed=true`` AND
        # the critic is frozen (the cache is invalid if the critic's
        # encoder changes mid-run). Single-rank: main process owns
        # the build; other ranks wait at the barrier and currently
        # don't share the cache (the relabel pass is also single-rank
        # in v1, so rank>0 never hits the lookup site).
        cache_state_embed = (
            parl_enabled
            and self.q_critic is not None
            and bool(OmegaConf.select(
                cfg, 'parl.cache_state_embed', default=True))
            and bool(OmegaConf.select(
                cfg, 'parl.freeze_critic', default=True))
        )
        if cache_state_embed and accelerator.is_main_process:
            cache_n_frames = int(getattr(dataset, 'n_frames_total', 0))
            if cache_n_frames > 0:
                cache_t, cache_v = self._precompute_state_embed_cache(
                    train_dataset=dataset,
                    val_dataset=val_dataset,
                    critic=self.q_critic,
                    n_frames=cache_n_frames,
                    device=device,
                    accelerator=accelerator,
                    parl_cfg=cfg.parl,
                )
                self._parl_state_embed_cache = cache_t
                self._parl_state_embed_valid = cache_v
        accelerator.wait_for_everyone()

        # PA-RL: optional one-time precompute of the proposer's
        # ``obs_encoder`` output for every dataset frame. Same idea
        # as the state_embed cache above but for the SECOND big
        # ViT-L the relabel pass would otherwise run (the proposer's
        # visual encoder inside ``propose_actions``). Combined, the
        # two caches eliminate ALL ViT-L forwards from the per-iter
        # relabel cost; the only remaining heavy compute is the
        # U-Net's K candidate diffusion samples plus the tiny
        # action/q-head MLPs.
        # Gated on (a) the proposer encoder actually being frozen
        # (``training.freeze_encoder=true``) and (b)
        # ``parl.cache_global_cond=true`` (default true). The proposer
        # we cache against is the EMA snapshot when available --
        # matches the proposer the relabel pass uses; the encoder
        # weights are identical to the underlying model's because
        # EMA only steps params that move and the encoder doesn't.
        cache_global_cond = (
            parl_enabled
            and bool(OmegaConf.select(
                cfg, 'parl.cache_global_cond', default=True))
            and bool(OmegaConf.select(
                cfg, 'training.freeze_encoder', default=False))
        )
        if cache_global_cond and accelerator.is_main_process:
            cache_n_frames = int(getattr(dataset, 'n_frames_total', 0))
            if cache_n_frames > 0:
                cache_proposer = (
                    self.ema_model if self.ema_model is not None
                    else accelerator.unwrap_model(self.model)
                )
                cache_proposer.eval()
                # Encoder must be in eval mode for deterministic
                # transforms, in particular to ensure dropout (if
                # any) is disabled. requires_grad is already false
                # via the freeze_encoder enforcement at the top of
                # the training loop, but eval() also matters for
                # batchnorm / dropout layers a future encoder might
                # carry.
                cache_proposer.obs_encoder.eval()
                cache_t, cache_v = self._precompute_global_cond_cache(
                    train_dataset=dataset,
                    val_dataset=val_dataset,
                    proposer=cache_proposer,
                    n_frames=cache_n_frames,
                    device=device,
                    accelerator=accelerator,
                    parl_cfg=cfg.parl,
                )
                self._parl_global_cond_cache = cache_t
                self._parl_global_cond_valid = cache_v
        accelerator.wait_for_everyone()

        # save batch for sampling
        train_sampling_batch = None

        # PA-RL relabel cache lives on disk for resumability + sanity
        # inspection ("did the optimized chunks actually move away from
        # the BC chunks?"). The in-memory swap into the train / val
        # datasets happens immediately afterwards so dataloader workers
        # spawned for the next epoch see the new data via the dataset
        # they hold a reference to.
        parl_relabel_every = int(OmegaConf.select(
            cfg, 'parl.relabel_every', default=1)) if parl_enabled else 0
        parl_save_cache = bool(OmegaConf.select(
            cfg, 'parl.save_cache_to_disk', default=True)) if parl_enabled else False
        # Pre-compute per-frame array shape so the relabel helper can
        # allocate without re-reading the replay buffer. ``dataset``
        # is the train split here; both train + val share the same
        # n_frames since they share the underlying ReplayBuffer.
        parl_n_frames = int(getattr(dataset, 'n_frames_total', 0))
        parl_action_horizon = int(getattr(dataset, 'action_horizon', 0))
        parl_action_dim = int(getattr(dataset, 'action_dim', 0))

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(self.epoch, cfg.training.num_epochs):
                # ========= PA-RL: optional relabel pass ==========
                # Runs at the start of each epoch where the cycle hits
                # (epoch 0 included, so we bootstrap from the BC ckpt's
                # weights). Single-rank: only the main process samples
                # / scores; other ranks wait at the barrier and then
                # all ranks load the freshly-written cache file.
                if (
                    parl_enabled
                    and self.q_critic is not None
                    and parl_relabel_every > 0
                    and (self.epoch % parl_relabel_every) == 0
                ):
                    parl_actions = None
                    parl_valid = None
                    parl_info = {}
                    if accelerator.is_main_process:
                        # Use EMA shadow as the proposer when available
                        # (more stable and matches inference); fall
                        # back to the unwrapped main model otherwise.
                        proposer_name = str(OmegaConf.select(
                            cfg, 'parl.proposer', default='ema'))
                        if proposer_name == 'ema' and self.ema_model is not None:
                            proposer = self.ema_model
                        else:
                            proposer = accelerator.unwrap_model(self.model)
                        accelerator.print(
                            f"[PA-RL] starting relabel for epoch={self.epoch} "
                            f"(proposer={proposer_name})"
                        )
                        parl_actions, parl_valid, parl_info = self._parl_relabel(
                            train_dataset=dataset,
                            val_dataset=val_dataset,
                            proposer=proposer,
                            critic=self.q_critic,
                            n_frames=parl_n_frames,
                            action_horizon=parl_action_horizon,
                            action_dim=parl_action_dim,
                            device=device,
                            accelerator=accelerator,
                            parl_cfg=cfg.parl,
                            state_embed_cache=self._parl_state_embed_cache,
                            state_embed_valid=self._parl_state_embed_valid,
                            global_cond_cache=self._parl_global_cond_cache,
                            global_cond_valid=self._parl_global_cond_valid,
                        )
                        if parl_save_cache:
                            cache_path = os.path.join(
                                self.output_dir,
                                f'relabel_cache_epoch{self.epoch:04d}.npy',
                            )
                            valid_path = os.path.join(
                                self.output_dir,
                                f'relabel_valid_epoch{self.epoch:04d}.npy',
                            )
                            np.save(cache_path, parl_actions)
                            np.save(valid_path, parl_valid)
                            accelerator.print(
                                f"[PA-RL] wrote cache: {cache_path} "
                                f"({parl_actions.nbytes / 1e6:.1f} MB) "
                                f"+ valid mask"
                            )
                    accelerator.wait_for_everyone()
                    # Multi-rank: ranks > 0 mmap the freshly-written
                    # cache from disk; rank 0 keeps its in-memory
                    # array (cheaper, no roundtrip). For the v1
                    # single-GPU path the rank > 0 branch is dead
                    # code -- but it costs nothing and future-proofs
                    # the multi-GPU path.
                    if accelerator.is_main_process:
                        dataset.set_action_optimized(parl_actions, parl_valid)
                        val_dataset.set_action_optimized(parl_actions, parl_valid)
                    elif parl_save_cache:
                        cache_path = os.path.join(
                            self.output_dir,
                            f'relabel_cache_epoch{self.epoch:04d}.npy',
                        )
                        valid_path = os.path.join(
                            self.output_dir,
                            f'relabel_valid_epoch{self.epoch:04d}.npy',
                        )
                        actions_mm = np.load(cache_path, mmap_mode='r')
                        valid_mm = np.load(valid_path, mmap_mode='r').astype(bool)
                        dataset.set_action_optimized(actions_mm, valid_mm)
                        val_dataset.set_action_optimized(actions_mm, valid_mm)
                    accelerator.wait_for_everyone()

                    # Rebuild dataloaders so any persistent_workers
                    # spawn fresh and pick up the swapped cache via
                    # the dataset reference. Re-prepare with
                    # accelerator so DistributedSampler etc. is
                    # re-applied for multi-GPU.
                    fresh_train_loader = DataLoader(dataset, **cfg.dataloader)
                    fresh_val_loader = DataLoader(val_dataset, **cfg.val_dataloader)
                    train_dataloader, val_dataloader = accelerator.prepare(
                        fresh_train_loader, fresh_val_loader,
                    )
                    if accelerator.is_main_process and parl_info:
                        accelerator.log(
                            {**parl_info, 'epoch': self.epoch},
                            step=self.global_step,
                        )

                self.model.train()

                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    # ``self.model`` is a DDP wrapper after
                    # accelerator.prepare(), so ``obs_encoder`` lives on
                    # the inner module. Use accelerator.unwrap_model so
                    # this works in both single-GPU (no wrapper) and
                    # multi-GPU DDP. eval() + requires_grad_(False)
                    # mutate state in place; DDP forwards the
                    # frozen-grad signal through the wrapper.
                    _unwrapped = accelerator.unwrap_model(self.model)
                    _unwrapped.obs_encoder.eval()
                    _unwrapped.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        
                        # always use the latest batch
                        train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(accelerator.unwrap_model(self.model))

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        # Log bbox mask channel weight norms. Sample frequently
                        # at the start so the early growth (from zero) is
                        # visible on the wandb curve, then less often once
                        # training is settled. Cheap (a few tensor norms) but
                        # we still gate by step to avoid spamming.
                        if accelerator.is_main_process:
                            log_every = (10 if self.global_step < 500 else 50)
                            if (self.global_step % log_every) == 0:
                                _unwrapped = accelerator.unwrap_model(self.model)
                                step_log.update(_bbox_mask_weight_metrics(_unwrapped))
                                step_log.update(_awr_metrics(_unwrapped))
                                step_log.update(_parl_metrics(_unwrapped))

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            accelerator.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # End-of-epoch snapshot of the mask channel weights. Always
                # emitted (no throttling) so the wandb x-axis "epoch" has a
                # clean per-epoch curve to chart alongside train_loss.
                if accelerator.is_main_process:
                    _unwrapped = accelerator.unwrap_model(self.model)
                    step_log.update(_bbox_mask_weight_metrics(_unwrapped))
                    step_log.update(_awr_metrics(_unwrapped))
                    step_log.update(_parl_metrics(_unwrapped))

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                # Runs on main rank only (other ranks idle through here, then
                # rejoin at accelerator.log below). Uses the unwrapped policy
                # (EMA if enabled) so it reflects what we'd actually deploy.
                if (self.epoch % cfg.training.val_every) == 0 and len(val_dataloader) > 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                # policy is already the unwrapped (EMA if
                                # enabled) module — its forward() returns the
                                # same diffusion loss the train step uses.
                                loss = policy(batch)
                                val_losses.append(loss.item())
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            step_log['val_loss'] = float(np.mean(val_losses))
                
                def _action_mse_components(pred, gt):
                    """Compute (full, pos, rot, width) MSE components from
                    (B, T, A) tensors viewed as (B, T, num_robot, 10).
                    """
                    B, T, _ = pred.shape
                    pred_v = pred.view(B, T, -1, 10)
                    gt_v = gt.view(B, T, -1, 10)
                    return (
                        torch.nn.functional.mse_loss(pred_v, gt_v),
                        torch.nn.functional.mse_loss(pred_v[..., :3], gt_v[..., :3]),
                        torch.nn.functional.mse_loss(pred_v[..., 3:9], gt_v[..., 3:9]),
                        torch.nn.functional.mse_loss(pred_v[..., 9], gt_v[..., 9]),
                    )

                def log_action_mse(step_log, category, pred_action, batch):
                    """Log action MSE components.

                    PA-RL changes this from the original "vs dataset action"
                    behavior. The dataset for fine-tuning contains failure
                    rollouts (misses, corrections) we explicitly do NOT want
                    to imitate; we are training the policy to imitate the
                    OPTIMIZED action targets emitted by the relabel pass.
                    Therefore the primary action_mse metric is computed
                    against ``batch['action_optimized']`` masked to samples
                    where ``has_action_optimized`` is set. The "vs dataset"
                    version is still logged under ``*_dataset`` because the
                    GAP between the two is the cleanest drift signal we
                    have without an expert-only val split:
                      * primary low + dataset stable: good progress.
                      * primary low + dataset rising: optimized targets
                        drifting from BC (turn anchor up, or critic has
                        blind spots).
                      * primary rising: distillation failing.

                    For categories without an optimized cache yet (e.g. on
                    the very first epoch before relabel runs, or with PA-RL
                    disabled), the primary metric falls back to the dataset
                    action so the metric remains continuous across modes.
                    """
                    gt_dataset = batch['action']
                    full_d, pos_d, rot_d, w_d = _action_mse_components(
                        pred_action, gt_dataset)
                    step_log[f'{category}_action_mse_error_dataset'] = full_d
                    step_log[f'{category}_action_mse_error_pos_dataset'] = pos_d
                    step_log[f'{category}_action_mse_error_rot_dataset'] = rot_d
                    step_log[f'{category}_action_mse_error_width_dataset'] = w_d

                    gt_opt = batch.get('action_optimized', None)
                    has_opt = batch.get('has_action_optimized', None)
                    use_opt = (
                        gt_opt is not None
                        and has_opt is not None
                        and bool((has_opt > 0.5).any().item())
                    )
                    if use_opt:
                        mask = (has_opt > 0.5).view(-1)
                        pred_m = pred_action[mask]
                        gt_m = gt_opt[mask]
                        full_o, pos_o, rot_o, w_o = _action_mse_components(
                            pred_m, gt_m)
                        # Primary monitor target: how well the deployed
                        # (full DDIM rollout) policy matches the actual
                        # training target, on the subset that has one.
                        step_log[f'{category}_action_mse_error'] = full_o
                        step_log[f'{category}_action_mse_error_pos'] = pos_o
                        step_log[f'{category}_action_mse_error_rot'] = rot_o
                        step_log[f'{category}_action_mse_error_width'] = w_o
                    else:
                        # Pre-relabel or non-PARL run: optimized cache not
                        # populated for this category. Use the dataset
                        # comparison as the primary metric so monitor_key
                        # stays well-defined and the wandb plot is
                        # continuous.
                        step_log[f'{category}_action_mse_error'] = full_d
                        step_log[f'{category}_action_mse_error_pos'] = pos_d
                        step_log[f'{category}_action_mse_error_rot'] = rot_d
                        step_log[f'{category}_action_mse_error_width'] = w_d
                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        pred_action = policy.predict_action(batch['obs'], None)['action_pred']
                        log_action_mse(step_log, 'train', pred_action, batch)

                        if len(val_dataloader) > 0:
                            val_sampling_batch = next(iter(val_dataloader))
                            batch = dict_apply(val_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            pred_action = policy.predict_action(batch['obs'], None)['action_pred']
                            log_action_mse(step_log, 'val', pred_action, batch)

                        del batch
                        del pred_action
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and accelerator.is_main_process:
                    # unwrap the model to save ckpt
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                    # recover the DDP model
                    self.model = model_ddp
                # ========= eval end for this epoch ==========
                # end of epoch
                # log of last step is combined with validation and rollout
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        accelerator.end_training()
