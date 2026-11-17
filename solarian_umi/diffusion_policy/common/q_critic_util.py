"""Load a trained IQL critic checkpoint into a runnable ``nn.Module``.

This is the single source of truth for "open the .ckpt produced by
``TrainIqlWorkspace`` and return a ready-to-use ``IQLCritic`` on the
target device". Three call sites delegate to it:

  * ``policy_server.load_q_critic`` (online inference: rerank /
    Q-guidance during eval)
  * ``zarr_q_v._load_iql_workspace`` (offline Q/V materialization
    that writes ``rl/q`` / ``rl/v`` back into the dataset zarr)
  * PA-RL training in
    ``TrainDiffusionUnetImageWorkspace`` (relabel pass that produces
    optimized action chunks for the diffusion DDPM target)

Keeping the load logic in one place is what guarantees that a Q value
seen during PA-RL relabel matches the Q value the policy server
observes at deployment time -- otherwise it would be very easy for a
subtle env / cfg-override drift to introduce a calibration gap that's
invisible until the policy actively learns to exploit it.
"""

from __future__ import annotations

import logging as _logging
import os
import time
from typing import Any, Optional, Tuple

import dill
import hydra
import torch
from omegaconf import OmegaConf, open_dict

# IQL workspace registers an ``${eval:...}`` resolver via its module
# import. Mirror it here in case this util is loaded BEFORE any IQL
# code path -- otherwise OmegaConf.select on a saved cfg blows up.
OmegaConf.register_new_resolver("eval", eval, replace=True)

_default_logger = _logging.getLogger(__name__)


def load_iql_critic(
    ckpt_path: str,
    device: torch.device,
    *,
    logger: Optional[_logging.Logger] = None,
) -> Tuple[torch.nn.Module, Any]:
    """Load an IQL critic checkpoint and return ``(critic.eval().to(device), cfg)``.

    Behavior matches the previous in-line implementations in
    ``zarr_q_v._load_iql_workspace`` and ``policy_server.load_q_critic``
    (which now delegate to this):

      * Reads the pickle with ``dill`` (the same pickler
        ``BaseWorkspace.save_checkpoint`` uses) onto CPU.
      * Forces ``iql.pretrained_obs_encoder_ckpt = None`` because the
        path was baked at training time (typically under ``/data/...``
        on the Modal volume) and likely does not exist locally. The
        following ``state_dict`` load restores the encoder weights
        anyway, so this is purely a "skip a download we don't need".
      * Leaves ``policy.obs_encoder.pretrained=True``: ``TimmObsEncoder``
        asserts ``frozen=True => pretrained=True`` and IQL defaults to
        ``freeze_visual=true``. The trained ``state_dict`` overwrites
        the timm weights a moment later, so this is at most a one-time
        HF-cache read.
      * Loads ``state_dicts['model']`` with ``strict=False`` and warns
        on missing / unexpected keys. A truly garbled ckpt should still
        fail downstream when ``q_values`` produces nonsense.
      * Calls ``.eval().to(device)`` on the critic. Callers can add a
        ``torch.cuda.synchronize()`` afterward if they care about
        accurate timing.

    Args:
        ckpt_path: path to the IQL ``.ckpt``. ``~`` is expanded.
        device: target device for the critic.
        logger: optional logger; defaults to a module-local logger so
            this util is silent under ``pytest`` / scripts that haven't
            configured logging.

    Returns:
        ``(critic, cfg)`` -- ``critic`` is the loaded ``IQLCritic``
        in ``eval()`` mode on ``device``; ``cfg`` is the full Hydra
        cfg snapshot the ckpt was trained with (callers reuse it to
        rebuild matching datasets / shape_meta).
    """
    log = logger or _default_logger
    ckpt_path = os.path.expanduser(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"IQL checkpoint not found: {ckpt_path}")

    log.info(f"Loading IQL checkpoint: {ckpt_path}")
    t0 = time.perf_counter()
    with open(ckpt_path, "rb") as f:
        payload = torch.load(f, map_location="cpu", pickle_module=dill)
    log.info(f"  read payload in {time.perf_counter() - t0:.2f}s")

    cfg = payload["cfg"]

    try:
        if OmegaConf.select(cfg, "iql.pretrained_obs_encoder_ckpt"):
            with open_dict(cfg):
                cfg.iql.pretrained_obs_encoder_ckpt = None
    except Exception as e:
        log.warning(f"could not override iql.pretrained_obs_encoder_ckpt: {e}")

    log.info(f"  workspace target: {cfg._target_}")
    cls = hydra.utils.get_class(cfg._target_)

    t0 = time.perf_counter()
    workspace = cls(cfg)
    log.info(f"  built workspace in {time.perf_counter() - t0:.2f}s")

    sds = payload.get("state_dicts", {})
    if "model" not in sds:
        raise RuntimeError(
            f"IQL checkpoint has no state_dicts['model']; "
            f"got keys {list(sds.keys())}"
        )

    missing, unexpected = workspace.model.load_state_dict(
        sds["model"], strict=False
    )
    if missing:
        log.warning(
            f"  load_state_dict: {len(missing)} missing keys "
            f"(first: {missing[:3]})"
        )
    if unexpected:
        log.warning(
            f"  load_state_dict: {len(unexpected)} unexpected keys "
            f"(first: {unexpected[:3]})"
        )

    model = workspace.model.eval().to(device)
    return model, cfg
