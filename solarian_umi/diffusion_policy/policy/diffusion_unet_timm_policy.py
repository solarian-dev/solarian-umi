# Adapted from Universal Manipulation Interface.
# Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab.
# https://github.com/real-stanford/universal_manipulation_interface
# Licensed under the MIT License. The full text is LICENSE.umi
# beside this package.

import copy
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from solarian_umi.diffusion_policy.model.common.normalizer import LinearNormalizer
from solarian_umi.diffusion_policy.policy.base_image_policy import BaseImagePolicy
from solarian_umi.diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from solarian_umi.diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from solarian_umi.diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from solarian_umi.diffusion_policy.common.pytorch_util import dict_apply


class DiffusionUnetTimmPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: TimmObsEncoder,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            input_pertub=0.1,
            inpaint_fixed_action_prefix=False,
            train_diffusion_n_samples=1,
            # ============== AWR finetuning knobs ==============
            # Advantage-Weighted Regression: weight each sample's
            # denoising MSE by ``exp(advantage / awr_beta)``. Disabled
            # (legacy BC) when awr_beta <= 0 OR the batch lacks
            # ``advantage`` / ``has_advantage`` keys (which UmiDataset
            # adds when rl/q + rl/v are present in the source zarr).
            #
            # awr_beta:    temperature on the advantage. Lower beta
            #              sharpens the weighting (more like Q-max but
            #              unstable); higher beta softens it toward BC.
            #              A sensible default is roughly ``std(A)`` on
            #              your dataset (your IQL ckpt name encodes
            #              ``val_adv_std``, so use that as a starting
            #              point). 0.0 disables AWR entirely.
            # awr_w_max:   per-sample weight clip BEFORE batch norm.
            #              Caps the influence of a single outlier with
            #              very large advantage. Typical 10..100.
            # awr_lambda:  blend between BC weight (1.0) and AWR weight
            #              ``w``. 1.0 is pure AWR; 0.0 collapses back
            #              to BC; values in (0, 1) act as a "safety
            #              belt" toward the BC checkpoint's behavior.
            # awr_normalize_weights: if True, divide weights by their
            #              batch mean before applying. Makes the
            #              effective LR roughly invariant to beta and
            #              to which subset of the data has high
            #              advantage. Strongly recommended.
            # awr_use_ema_weights: deprecated knob slot kept here as a
            #              future hook (e.g. for a slow-moving running
            #              mean of the batch weight to stabilize early
            #              training); currently unused.
            awr_beta=0.0,
            awr_w_max=20.0,
            awr_lambda=1.0,
            awr_normalize_weights=True,
            # ============== BC-anchored AWR knobs ==============
            # The plain AWR loss above can overfit to high-advantage
            # outliers and drift the policy off the BC manifold,
            # producing twitchy / unstable rollouts. The three extra
            # knobs below let us anchor the fine-tune back toward BC:
            #
            #   awr_w_min:          lower clamp on per-sample weights
            #                       AFTER the log-cap. Prevents the
            #                       optimizer from completely ignoring
            #                       low-advantage but on-distribution
            #                       samples (the prior says "still do
            #                       what BC said to do"). Typical
            #                       0.3..0.5; 0 reproduces legacy AWR.
            #   awr_bc_lambda:      coefficient on the UNWEIGHTED BC
            #                       loss term (plain denoising MSE,
            #                       SAME pred/target as the AWR term so
            #                       there's no extra forward). Pulls
            #                       the policy back toward "imitate
            #                       every demo equally". Typical
            #                       0.5..5.0; 0 disables the term.
            #   awr_anchor_lambda:  coefficient on the FROZEN-REFERENCE
            #                       anchor: MSE between the current
            #                       policy's eps-prediction and a
            #                       frozen copy of the BC policy's
            #                       eps-prediction at the same
            #                       (x_t, t, obs). Acts as a soft KL
            #                       regularizer toward the BC policy
            #                       in eps-space. Requires the
            #                       workspace to call
            #                       ``init_frozen_ref()`` once after
            #                       loading ``init_from_ckpt``.
            #                       Typical 0.1..1.0; 0 disables.
            awr_w_min=0.0,
            awr_bc_lambda=0.0,
            awr_anchor_lambda=0.0,
            # parameters passed to step
            **kwargs
        ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        # get feature dim
        obs_feature_dim = np.prod(obs_encoder.output_shape())


        # create diffusion model
        assert obs_as_global_cond
        input_dim = action_dim
        global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon # used for training
        self.obs_as_global_cond = obs_as_global_cond
        self.input_pertub = input_pertub
        self.inpaint_fixed_action_prefix = inpaint_fixed_action_prefix
        self.train_diffusion_n_samples = int(train_diffusion_n_samples)
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # AWR finetuning state. ``last_awr_info`` carries per-batch
        # diagnostic scalars (weight stats, fraction of zero-advantage
        # samples, etc.) that the workspace pulls into wandb. Set to
        # None when AWR is disabled or the batch had no advantage info.
        self.awr_beta = float(awr_beta)
        self.awr_w_max = float(awr_w_max)
        self.awr_lambda = float(awr_lambda)
        self.awr_normalize_weights = bool(awr_normalize_weights)
        self.awr_w_min = float(awr_w_min)
        self.awr_bc_lambda = float(awr_bc_lambda)
        self.awr_anchor_lambda = float(awr_anchor_lambda)
        # Frozen reference U-Net for the anchor term. Materialized
        # lazily by `init_frozen_ref()` (called by the workspace AFTER
        # `init_from_ckpt` has loaded the BC weights into self.model).
        # Excluded from `state_dict()` so AWR checkpoints don't double
        # in size (the ref is re-built from `init_from_ckpt` on the
        # next finetune; for inference loads it's never needed).
        self.frozen_ref_model: Optional[nn.Module] = None
        self.last_awr_info: Dict[str, float] = {}

    def init_frozen_ref(self) -> None:
        """Snapshot the current self.model as a frozen anchor.

        Called by the workspace right after ``init_from_ckpt`` has
        loaded the BC weights, so the frozen reference IS the BC
        policy at finetune-step-0. No-op when the anchor term is
        disabled (``awr_anchor_lambda <= 0``) so the legacy AWR run
        doesn't pay the extra ~600 MB of GPU memory for a copy of
        the U-Net it'll never use.
        """
        if self.awr_anchor_lambda <= 0.0:
            return
        if self.frozen_ref_model is not None:
            return  # idempotent
        # Deep-copy ensures parameter tensors don't share storage with
        # self.model -- otherwise an in-place op on self.model (or
        # accelerator-wrapping) could mutate the anchor.
        ref = copy.deepcopy(self.model).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        # We register as an attribute (not a submodule) so it is NOT
        # part of `self.parameters()` and therefore not picked up by
        # the optimizer / DDP / EMA. The eval-mode forward is invoked
        # explicitly from compute_loss.
        object.__setattr__(self, 'frozen_ref_model', ref)

    def state_dict(self, *args, **kwargs):
        """Strip the frozen reference from saved checkpoints.

        Including it would double the on-disk size of every AWR
        checkpoint for zero benefit (the reference is rebuilt from
        ``init_from_ckpt`` on the next finetune run, and inference
        loads don't need it at all).
        """
        sd = super().state_dict(*args, **kwargs)
        keys_to_drop = [k for k in sd.keys() if k.startswith('frozen_ref_model.')]
        for k in keys_to_drop:
            del sd[k]
        return sd

    def load_state_dict(self, state_dict, strict: bool = True, *args, **kwargs):
        """Tolerate the absence of frozen_ref_model.* keys on load.

        AWR checkpoints (which we save via the overridden
        ``state_dict`` above) never contain those keys, but a
        downstream ``init_frozen_ref()`` re-materializes the buffer
        from current weights so nothing is lost.
        """
        if strict:
            ref_keys = [k for k in self.state_dict().keys()
                        if k.startswith('frozen_ref_model.')]
            if ref_keys and not any(
                    k.startswith('frozen_ref_model.') for k in state_dict.keys()):
                # Demote strict load for this specific case.
                strict = False
        return super().load_state_dict(state_dict, strict=strict, *args, **kwargs)

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data,
            condition_mask,
            local_cond=None,
            global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
        ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], fixed_action_prefix: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        fixed_action_prefix: unnormalized action prefix
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        # condition through global feature
        global_cond = self.obs_encoder(nobs)

        # empty data for action
        cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            cond_data[:, :n_fixed_steps] = fixed_action_prefix
            cond_mask[:, :n_fixed_steps] = True
            cond_data = self.normalizer['action'].normalize(cond_data)


        # run sampling
        nsample = self.conditional_sample(
            condition_data=cond_data, 
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(nsample)
        
        result = {
            'action': action_pred,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions_dataset = self.normalizer['action'].normalize(batch['action'])

        # ---------------- PA-RL action-optimization target ----------------
        # When the dataloader emits ``action_optimized`` (UmiDataset's
        # cache populated by the workspace's relabel pass), swap the
        # DDPM target from the dataset action to the optimized action
        # on a PER-SAMPLE basis. ``has_action_optimized`` is the gate
        # (1.0 => optimized; 0.0 => dataset fallback), mirroring the
        # ``has_advantage`` convention so partially-cached batches don't
        # silently zero out a fraction of samples.
        #
        # PA-RL is mutually exclusive with AWR within a single batch:
        # we don't want to "Q-optimize the action then re-weight the
        # loss on it by stale dataset advantages". When PA-RL fires,
        # the AWR knobs are forced inactive and the loss reduces to
        # the standard DDPM denoising loss against the optimized chunk
        # plus any optional anchor terms. The PA-RL config sets
        # ``awr_beta=0`` for the same reason; this gate is just defense
        # in depth.
        action_optimized = batch.get('action_optimized', None)
        has_action_optimized = batch.get('has_action_optimized', None)
        parl_active = (
            action_optimized is not None
            and has_action_optimized is not None
            and bool((has_action_optimized > 0.5).any().item())
        )
        if parl_active:
            nactions_opt = self.normalizer['action'].normalize(action_optimized)
            has_opt = has_action_optimized.view(-1, 1, 1).to(
                dtype=torch.bool, device=nactions_opt.device)
            nactions = torch.where(has_opt, nactions_opt, nactions_dataset)
        else:
            nactions = nactions_dataset

        # AWR per-sample inputs. Pulled here (BEFORE the optional
        # train_diffusion_n_samples expansion below) so we can repeat
        # them in lockstep with the action / obs duplication. Both
        # default to None when the batch doesn't carry advantage info,
        # which is also the path taken when awr_beta <= 0.
        # When PA-RL is active we force these to None so the AWR
        # weighting path is fully off (see comment block above).
        advantage = batch.get('advantage', None) if not parl_active else None
        has_advantage = batch.get('has_advantage', None) if not parl_active else None
        awr_active = (
            self.awr_beta > 0.0
            and advantage is not None
            and has_advantage is not None
        )

        assert self.obs_as_global_cond
        global_cond = self.obs_encoder(nobs)

        # train on multiple diffusion samples per obs
        if self.train_diffusion_n_samples != 1:
            # repeat obs features and actions multiple times along the batch dimension
            # each sample will later have a different noise sample, effecty training 
            # more diffusion steps per each obs encoder forward pass
            global_cond = torch.repeat_interleave(global_cond, 
                repeats=self.train_diffusion_n_samples, dim=0)
            nactions = torch.repeat_interleave(nactions, 
                repeats=self.train_diffusion_n_samples, dim=0)
            if awr_active:
                advantage = torch.repeat_interleave(
                    advantage, repeats=self.train_diffusion_n_samples, dim=0)
                has_advantage = torch.repeat_interleave(
                    has_advantage, repeats=self.train_diffusion_n_samples, dim=0)

        trajectory = nactions
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # input perturbation by adding additonal noise to alleviate exposure bias
        # reference: https://github.com/forever208/DDPM-IP
        noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (nactions.shape[0],), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)
        
        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            timesteps, 
            local_cond=None,
            global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.type(loss.dtype)
        # Reduce to a single scalar per sample (B,). We keep this
        # un-meaned so AWR can apply its per-sample weight cleanly.
        loss_per_sample = reduce(loss, 'b ... -> b', 'mean')

        # Compute the anchor loss once (independent of AWR weighting)
        # so we can add it to either the BC-only or the AWR path.
        # Skipped when no frozen reference is attached, i.e. when
        # `awr_anchor_lambda` is 0 or `init_frozen_ref()` was never
        # called -- preserving byte-identical legacy BC behavior.
        loss_anchor = None
        if (
            self.awr_anchor_lambda > 0.0
            and self.frozen_ref_model is not None
        ):
            # The frozen ref is intentionally registered via
            # `object.__setattr__` (NOT as an nn.Module submodule) so
            # it's invisible to DDP / EMA / the optimizer. The
            # downside: `accelerator.prepare(self.model)` / `.to()`
            # don't recurse into it, so on the first forward it may
            # still be on CPU while the rest of the model is on the
            # current process's CUDA device. Move it ONCE here --
            # idempotent on subsequent steps and free when the device
            # already matches.
            ref_device = next(
                self.frozen_ref_model.parameters()).device
            if ref_device != noisy_trajectory.device:
                self.frozen_ref_model.to(noisy_trajectory.device)
            with torch.no_grad():
                pred_ref = self.frozen_ref_model(
                    noisy_trajectory, timesteps,
                    local_cond=None, global_cond=global_cond,
                )
            loss_anchor = F.mse_loss(pred, pred_ref)

        if not awr_active:
            # Legacy BC path: byte-identical to the pre-AWR loss
            # (mean over the per-sample MSE scalar is mathematically
            # equivalent to mean over the full (b, h, a) tensor).
            # Still permit the anchor term to fire here so
            # `awr_beta=0, awr_anchor_lambda>0` gives plain BC-with-
            # anchor (useful as a sanity baseline).
            self.last_awr_info = {}
            total = loss_per_sample.mean()
            if loss_anchor is not None:
                total = total + self.awr_anchor_lambda * loss_anchor
                self.last_awr_info = {
                    'awr/loss_bc': float(loss_per_sample.mean()),
                    'awr/loss_anchor': float(loss_anchor),
                }
            # PA-RL diagnostics: track what fraction of the batch was
            # sourced from the optimized cache vs fell back to dataset
            # actions. Mean L2 between optimized and dataset target
            # tells us whether the PA-RL relabel pass actually moved
            # actions (vs collapsed back to the BC chunk).
            if parl_active:
                with torch.no_grad():
                    has_opt_b = has_action_optimized.float().view(-1)
                    n_total = int(has_opt_b.numel())
                    frac_opt = float(has_opt_b.mean().item()) \
                        if n_total > 0 else 0.0
                    if has_opt_b.bool().any():
                        delta = (
                            nactions_opt[has_opt_b.bool()]
                            - nactions_dataset[has_opt_b.bool()]
                        ).flatten(start_dim=1)
                        l2 = float(delta.norm(dim=-1).mean().item())
                    else:
                        l2 = 0.0
                self.last_awr_info.update({
                    'parl/active': 1.0,
                    'parl/frac_optimized': frac_opt,
                    'parl/target_l2_change_norm_mean': l2,
                    'parl/loss_ddpm': float(loss_per_sample.mean().item()),
                    'parl/loss_anchor': (
                        float(loss_anchor.item())
                        if loss_anchor is not None else 0.0
                    ),
                })
            return total

        # ===== AWR weighting =====
        # advantage / has_advantage are detached float tensors from the
        # data loader, but be paranoid about dtype/device alignment in
        # case a future loader change supplies them in fp16 / cpu.
        adv = advantage.detach().to(
            device=loss_per_sample.device, dtype=loss_per_sample.dtype)
        m = has_advantage.detach().to(
            device=loss_per_sample.device, dtype=loss_per_sample.dtype)

        # Numerically safe exp: clamp the exponent so a single sample
        # with a very large advantage can't blow up to inf. We clamp
        # in *log space* against awr_w_max so the post-clamp weight is
        # exactly the same as `clamp(exp(...), max=awr_w_max)` but we
        # never materialize the inf intermediate.
        log_w_cap = float(np.log(self.awr_w_max))
        exponent = (adv / self.awr_beta).clamp(max=log_w_cap)
        w_awr = torch.exp(exponent)              # (B,), in (0, awr_w_max]

        # Optional lower clamp on the per-sample weight, applied BEFORE
        # the "no advantage info -> w=1" fallback so low-advantage
        # but on-distribution samples can't be silently zeroed by a
        # very low (or very negative) `adv / beta`. Acts as a soft
        # "always respect at least this much of BC" prior.
        if self.awr_w_min > 0.0:
            w_awr = w_awr.clamp(min=self.awr_w_min)

        # Replace AWR weight with 1.0 (i.e. plain BC) for samples that
        # were flagged as not having a valid IQL anchor by the dataset.
        # This keeps gradient information flowing through those samples
        # rather than zeroing them out, which would shrink the
        # effective batch size on partially-annotated datasets.
        w = torch.where(m > 0.5, w_awr, torch.ones_like(w_awr))

        # Optional per-batch normalization. Without it, the absolute
        # magnitude of the loss (and therefore the effective learning
        # rate) drifts with awr_beta and with the empirical advantage
        # distribution of the current batch. We intentionally normalize
        # over the FULL batch (including the BC-fallback samples) so
        # the relative weighting between AWR-eligible and ineligible
        # samples is preserved.
        if self.awr_normalize_weights:
            w = w / (w.mean() + 1e-8)

        # BC↔AWR linear blend. lambda=1 = pure AWR; lambda=0 collapses
        # to plain BC (useful as a sanity baseline at config flip).
        eff_w = (1.0 - self.awr_lambda) * torch.ones_like(w) \
                + self.awr_lambda * w

        # The "AWR" term in our 3-term BC-anchored loss.
        loss_awr = (loss_per_sample * eff_w).mean()

        # Optional UNWEIGHTED BC term ("pull back toward plain
        # imitation of every demo equally"). Uses the SAME per-sample
        # MSE we already computed -- no extra forward pass.
        loss_bc = loss_per_sample.mean()

        total = loss_awr
        if self.awr_bc_lambda > 0.0:
            total = total + self.awr_bc_lambda * loss_bc
        if loss_anchor is not None:
            total = total + self.awr_anchor_lambda * loss_anchor

        # Per-batch diagnostics. Stashed on the module so the workspace
        # can pull them into wandb without changing compute_loss's
        # contract (still returns a single scalar).
        with torch.no_grad():
            valid = m > 0.5
            n_valid = valid.sum().clamp_min(1)
            adv_valid_mean = (adv * valid).sum() / n_valid
            adv_valid_max = adv[valid].max() if valid.any() else torch.tensor(
                float('nan'), device=adv.device)
            adv_valid_min = adv[valid].min() if valid.any() else torch.tensor(
                float('nan'), device=adv.device)
            self.last_awr_info = {
                'awr/active': 1.0,
                'awr/beta': float(self.awr_beta),
                'awr/lambda': float(self.awr_lambda),
                'awr/bc_lambda': float(self.awr_bc_lambda),
                'awr/anchor_lambda': float(self.awr_anchor_lambda),
                'awr/w_min_floor': float(self.awr_w_min),
                'awr/w_mean': float(w.mean()),
                'awr/w_std': float(w.std()),
                'awr/w_min': float(w.min()),
                'awr/w_max_seen': float(w.max()),
                'awr/eff_w_mean': float(eff_w.mean()),
                'awr/eff_w_std': float(eff_w.std()),
                'awr/frac_clipped': float(
                    (exponent >= log_w_cap - 1e-6).float().mean()),
                'awr/frac_floored': (
                    float((w_awr <= self.awr_w_min + 1e-6).float().mean())
                    if self.awr_w_min > 0.0 else 0.0),
                'awr/frac_has_adv': float(valid.float().mean()),
                'awr/adv_valid_mean': float(adv_valid_mean),
                'awr/adv_valid_min': float(adv_valid_min),
                'awr/adv_valid_max': float(adv_valid_max),
                'awr/loss_awr': float(loss_awr),
                'awr/loss_bc': float(loss_bc),
                'awr/loss_anchor': (
                    float(loss_anchor) if loss_anchor is not None else 0.0),
            }

        return total

    def forward(self, batch):
        return self.compute_loss(batch)