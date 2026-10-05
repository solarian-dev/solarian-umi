"""PA-RL (Policy-Agnostic RL) action optimization.

Implements the action-optimization stage of PA-RL (Sobol Mark et al.,
2024, "Policy Agnostic RL", arXiv:2412.06685):

  1. Propose K action chunks per state by sampling from the current
     diffusion policy.
  2. Global optimization: rerank by the IQL critic and keep the top-m.
  3. Local optimization: T gradient ascent steps on each kept chunk
     w.r.t. min(Q1, Q2), with a per-step revert-on-no-improve guard.
  4. Pick: argmax-Q or one sample from softmax(Q / tau) over the
     optimized chunks (paper's recommendation: argmax when the
     intra-batch Q-variance is small, softmax otherwise; a single
     ``mode`` knob picks per call).

The whole pipeline is a *pure function over a frozen IQL critic*: no
parameter updates, no global state. That makes it equally usable from

  * the relabel pass in ``TrainDiffusionUnetImageWorkspace`` (offline
    PA-RL distillation), and
  * future inference-time online PA-RL, should we add it.

Performance notes:
  * The visual encoder is the dominant cost. We compute
    ``critic.state_embed(obs)`` ONCE per batch of states and reuse it
    for both global rerank and every step of local gradient ascent
    (mirrors ``policy_server._q_min_from_state_embed``).
  * Chunk sampling expands obs along a fake "K" axis so the diffusion
    policy's U-Net sees a single batched tensor of shape
    ``(B*K, H, A)``. The encoder runs once (with B*K rows of repeated
    obs); the U-Net's per-sample noise gives the K candidates their
    diversity.
  * Local optimization runs on UNNORMALIZED actions (the critic owns
    its own ``LinearNormalizer`` and re-normalizes inside
    ``action_embed``). This keeps the gradient direction interpretable
    in robot-space units, which matters because the Q-net was trained
    on the same unnormalized action representation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Q-min helper: avoid re-running the IQL obs encoder per candidate.
# ---------------------------------------------------------------------------


def q_min_from_state_embed(
    critic,
    state_embed: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    """``min(Q1, Q2)`` given a precomputed ``state_embed`` and an
    UNNORMALIZED action chunk.

    Identical contract to ``policy_server._q_min_from_state_embed``;
    duplicated here so this module has no dependency on the FastAPI
    server. ``critic.action_embed(...)`` re-normalizes internally.

    Shapes:
        state_embed: (N, D)
        action:      (N, H, A)  -- unnormalized
    Returns:
        (N,) tensor.
    """
    a = critic.action_embed(action, target=False)
    sa = torch.cat([state_embed, a], dim=-1)
    q1 = critic.q1_head(sa).squeeze(-1)
    q2 = critic.q2_head(sa).squeeze(-1)
    return torch.minimum(q1, q2)


# ---------------------------------------------------------------------------
# Stage I.a: proposal — sample K chunks per state from the diffusion policy.
# ---------------------------------------------------------------------------


@torch.no_grad()
def propose_actions(
    proposer,
    obs_dict: Dict[str, torch.Tensor],
    K: int,
    *,
    inference_steps: Optional[int] = None,
    global_cond_override: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample K UNNORMALIZED action chunks per state.

    Implements Equation 4.1's "candidate set"
    :math:`\\mathcal{A}_{\\pi_\\varphi, K}(s)` for diffusion policies.

    Performance note: the visual encoder (typically a frozen ViT-L)
    is the dominant cost of this call. Naively expanding the obs to
    ``(B*K, ...)`` before the encoder runs the (heavy, O(image)) ViT
    K times on each unique state. We instead:

      1. encode the B UNIQUE obs once -> ``global_cond_B: (B, D)``
      2. repeat-interleave the encoded vector to ``(B*K, D)``
      3. run the (lightweight, O(traj)) U-Net at batch B*K with K
         independent noise draws

    This is mathematically identical to the naive path because the
    encoder is row-independent, but cuts encoder cost K-fold (32x
    at the paper's default K=32). The U-Net cost is unchanged
    because each candidate still gets its own noise sample at every
    DDIM step.

    Args:
        proposer: a ``DiffusionUnetTimmPolicy``-shaped object exposing
            ``normalizer``, ``obs_encoder``, ``action_horizon``,
            ``action_dim``, ``device``, ``dtype``,
            ``num_inference_steps``, and ``conditional_sample`` /
            ``noise_scheduler`` / ``model``. The on-policy snapshot
            (e.g. EMA model) the relabel pass uses.
        obs_dict: ``(B, ...)`` per key, raw (unnormalized) obs.
        K: number of chunks per state.
        inference_steps: temporary override of
            ``proposer.num_inference_steps`` for the duration of this
            call. Useful to run cheaper sampling during relabel
            (e.g. 5 DDIM steps) than the policy uses at deployment
            (e.g. 16). Mutating the attribute is safe because we
            ALWAYS restore the original in a try/finally and we hold
            ``torch.no_grad`` so no autograd state is tied to the
            attribute. Set to ``None`` to use ``proposer.num_inference_steps``.
        global_cond_override: optional precomputed ``(B, D_cond)``
            tensor produced by the SAME ``proposer.obs_encoder`` for
            the SAME ``obs_dict``. When provided, we skip both the
            normalize and obs_encoder forwards entirely. Intended for
            workflows that pre-compute and cache the encoder output
            once for all dataset frames (the proposer's encoder is
            frozen when ``training.freeze_encoder=True``, so its
            output is invariant across relabel cycles). Must be on
            the same device as the U-Net; dtype is preserved.

    Returns:
        ``(B, K, H, A)`` tensor on ``proposer.device``, UNNORMALIZED
        (matches the contract of ``policy.predict_action``).
    """
    if K <= 0:
        raise ValueError(f"K must be >= 1, got {K}")

    B = next(iter(obs_dict.values())).shape[0]

    if global_cond_override is not None:
        # Cache hit: skip normalize + obs_encoder. Validate shape.
        if global_cond_override.shape[0] != B:
            raise ValueError(
                f"global_cond_override has B="
                f"{global_cond_override.shape[0]} but obs_dict has B={B}"
            )
        global_cond_B = global_cond_override
    else:
        nobs = proposer.normalizer.normalize(obs_dict)
        # Encode once on the B unique obs (NOT B*K) — the K candidates
        # per state share identical visual conditioning; only the U-Net
        # noise differs across replicas. Saves K-fold compute on the
        # visual encoder, which is the dominant relabel cost.
        global_cond_B = proposer.obs_encoder(nobs)  # (B, D_cond)
    if K > 1:
        global_cond = global_cond_B.repeat_interleave(K, dim=0)
    else:
        global_cond = global_cond_B

    saved_steps = proposer.num_inference_steps
    if inference_steps is not None:
        proposer.num_inference_steps = int(inference_steps)
    try:
        cond_data = torch.zeros(
            size=(B * K, proposer.action_horizon, proposer.action_dim),
            device=proposer.device, dtype=proposer.dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        nsample = proposer.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
        )
    finally:
        proposer.num_inference_steps = saved_steps

    actions = proposer.normalizer["action"].unnormalize(nsample)
    # (B*K, H, A) -> (B, K, H, A) without copying when K=1.
    actions = actions.view(B, K, proposer.action_horizon, proposer.action_dim)
    return actions


# ---------------------------------------------------------------------------
# Stage I.b: global optimization — rerank K candidates, keep top-m.
# ---------------------------------------------------------------------------


@torch.no_grad()
def global_optimize(
    critic,
    state_embed_K: torch.Tensor,
    candidates: torch.Tensor,
    m: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Equation 4.1: rerank K candidates per state and keep top-m.

    Args:
        critic: IQL critic (provides ``action_embed``, ``q1_head``,
            ``q2_head``).
        state_embed_K: (B*K, D) state embedding, one row per
            candidate. Caller must already have repeated the per-state
            embedding K times (so the critic doesn't re-encode obs per
            candidate).
        candidates: (B, K, H, A) UNNORMALIZED chunks.
        m: number of top candidates to retain. ``m <= K`` enforced.

    Returns:
        ``(top_actions, top_q)`` where
            top_actions: (B, m, H, A)
            top_q:       (B, m)  -- min(Q1, Q2) of each kept chunk.
    """
    B, K, H, A = candidates.shape
    if m <= 0:
        raise ValueError(f"m must be >= 1, got {m}")
    if m > K:
        raise ValueError(f"m ({m}) > K ({K})")

    flat = candidates.reshape(B * K, H, A)
    q_min = q_min_from_state_embed(critic, state_embed_K, flat)
    q_min = q_min.view(B, K)

    # ``torch.topk`` returns the m highest values and their indices
    # along the K dim. The sort is stable enough for our purposes
    # (ties are vanishingly unlikely on float32 Q).
    top_q, top_idx = q_min.topk(k=m, dim=1, largest=True, sorted=True)
    # Gather the corresponding chunks. expand+gather avoids the
    # advanced-indexing copy.
    idx_exp = top_idx.view(B, m, 1, 1).expand(B, m, H, A)
    top_actions = candidates.gather(dim=1, index=idx_exp).contiguous()
    return top_actions, top_q


# ---------------------------------------------------------------------------
# Stage I.c: local optimization — T grad ascent steps with revert-on-no-improve.
# ---------------------------------------------------------------------------


def local_optimize(
    critic,
    state_embed: torch.Tensor,
    actions: torch.Tensor,
    *,
    T: int,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Algorithm 1: T steps of gradient ascent on Q w.r.t. action.

    For each (state, candidate) pair we do up to T steps of
    ``a_{j+1} = a_j + alpha * grad_a Q(s, a_j)``. After each step we
    re-evaluate Q; if it did NOT improve, we REVERT to the previous
    action and stop early for that candidate (matching the
    "revert-on-no-improve" rule in Algorithm 1 lines 6-9 -- "we only
    execute a local update if it increases the Q-value", Section 4.3).

    Crucially: we do NOT re-run the obs encoder during this loop. The
    state embedding is computed once by the caller and passed in,
    while ``critic.action_embed`` (a tiny MLP over the flattened
    chunk) runs each step.

    Args:
        critic: frozen IQL critic.
        state_embed: (N, D)  -- one state-embed per candidate.
            Caller is responsible for replicating per-state embeds to
            match the candidate count.
        actions: (N, H, A) UNNORMALIZED candidates. Returned tensor
            has the same shape; gradient flow does NOT escape this
            function.
        T: max number of gradient steps per candidate.
        alpha: gradient step size in unnormalized action units. The
            paper uses 3e-4 for state-based tasks; image-based PA-RL
            uses the same default.

    Returns:
        ``(actions_opt, q_after, info)`` where
            actions_opt: (N, H, A)
            q_after:     (N,)         -- final ``min(Q1, Q2)``.
            info: aggregate diagnostics (q_before mean, q_after mean,
                  q_gain mean, frac_local_reverted, action_l2_change).
    """
    if actions.dim() != 3:
        raise ValueError(
            f"local_optimize expects (N, H, A) actions; got {actions.shape}")
    N = actions.shape[0]
    info: Dict[str, float] = {
        "n_candidates": int(N),
        "T": int(T),
        "alpha": float(alpha),
    }

    if N == 0 or T <= 0 or alpha <= 0.0:
        with torch.no_grad():
            q_now = q_min_from_state_embed(critic, state_embed, actions)
        info.update({
            "q_before_mean": float(q_now.mean().item()),
            "q_after_mean": float(q_now.mean().item()),
            "q_gain_mean": 0.0,
            "frac_local_reverted": 0.0,
            "action_l2_change_mean": 0.0,
        })
        return actions.detach(), q_now.detach(), info

    # Gradient ascent. We maintain a "current best" copy and only
    # replace it on a Q increase; this is the exact rule in Algorithm 1
    # which the paper treats as the safe knob ("not guaranteed to
    # improve... we only execute a local update if it increases the
    # Q-value", Sec 4.3).
    actions_curr = actions.detach().clone()
    with torch.no_grad():
        q_before = q_min_from_state_embed(critic, state_embed, actions_curr)
    q_curr = q_before.clone()

    # Track which candidates are still "live". Once a candidate
    # reverts, we freeze it for the rest of the loop (Algorithm 1's
    # explicit `Break`).
    alive = torch.ones(N, dtype=torch.bool, device=actions.device)
    n_steps_used = torch.zeros(N, dtype=torch.long, device=actions.device)

    for step_idx in range(int(T)):
        if not alive.any():
            break
        # Build a candidate tensor with grads only on the live rows;
        # autograd is happy if the dead rows simply don't contribute
        # to the loss we sum over.
        with torch.enable_grad():
            a_var = actions_curr.detach().clone().requires_grad_(True)
            q_now = q_min_from_state_embed(critic, state_embed, a_var)
            # Sum only the live rows so we don't compute dead-row grads.
            (q_now * alive.float()).sum().backward()
            grad = a_var.grad
        # Per-candidate proposed update. Dead rows get zeroed in-place
        # so they don't contaminate the next iteration.
        proposed = actions_curr + float(alpha) * grad.detach()
        # Re-eval Q at the proposed actions. ``no_grad`` is fine; we
        # just need the scalar comparison.
        with torch.no_grad():
            q_proposed = q_min_from_state_embed(critic, state_embed, proposed)
        improved = (q_proposed > q_curr) & alive
        # Where improved: accept; else: revert (keep actions_curr,
        # q_curr) and mark dead for subsequent steps.
        if improved.any():
            improved_exp = improved.view(N, 1, 1)
            actions_curr = torch.where(
                improved_exp, proposed, actions_curr,
            )
            q_curr = torch.where(improved, q_proposed, q_curr)
            n_steps_used = torch.where(
                improved, n_steps_used + 1, n_steps_used,
            )
        # Mark non-improved (and previously-alive) rows dead.
        alive = alive & improved

    # Final stats.
    with torch.no_grad():
        delta = (actions_curr - actions).flatten(start_dim=1)
        l2 = delta.norm(dim=-1)
        # A candidate is "reverted" if it didn't improve at step 0,
        # i.e. it took zero accepted steps and Q is unchanged.
        reverted = (n_steps_used == 0)
    info.update({
        "q_before_mean": float(q_before.mean().item()),
        "q_after_mean": float(q_curr.mean().item()),
        "q_gain_mean": float((q_curr - q_before).mean().item()),
        "frac_local_reverted": float(reverted.float().mean().item()),
        "action_l2_change_mean": float(l2.mean().item()),
        "n_steps_mean": float(n_steps_used.float().mean().item()),
        "n_steps_max": int(n_steps_used.max().item()),
    })
    return actions_curr.detach(), q_curr.detach(), info


# ---------------------------------------------------------------------------
# Stage II: pick — argmax or softmax(Q/tau) over the optimized set.
# ---------------------------------------------------------------------------


@dataclass
class PickResult:
    """Per-state pick output from ``pick_optimized``."""
    action: torch.Tensor   # (B, H, A)  -- the chosen (and possibly
                           # dataset) action for each state.
    chosen_q: torch.Tensor  # (B,)       -- Q value at the chosen action.
    chose_dataset: torch.Tensor  # (B,) bool -- True iff the dataset
                                 # action was selected for this state.
    info: Dict[str, Any]


def pick_optimized(
    critic,
    state_embed: torch.Tensor,
    optimized_actions: torch.Tensor,
    optimized_q: torch.Tensor,
    *,
    mode: str = "argmax",
    softmax_temperature: float = 1.0,
    dataset_action: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> PickResult:
    """Equation 4.3: select ONE action per state from the optimized set.

    Args:
        critic: frozen IQL critic. Used only when ``dataset_action`` is
            provided, to compute its Q value and merge it into the
            candidate set.
        state_embed: (B, D) one row per state. NOT replicated.
        optimized_actions: (B, m, H, A) the post-local-opt actions.
        optimized_q: (B, m)               their min(Q1, Q2) values.
        mode: ``"argmax"`` returns ``argmax_a Q(s, a)``;
              ``"softmax"`` samples a single action per state from
              ``Categorical(logits=Q / tau)``.
        softmax_temperature: tau in the softmax mode. Smaller -> closer
            to argmax; larger -> closer to uniform.
        dataset_action: optional (B, H, A) dataset action chunk to
            include in the candidate set ("we also find that
            additionally including the action a appearing at a given
            state in the dataset into action optimization can sometimes
            be helpful." -- Sec 4.3, Implementation details). If
            provided we evaluate its Q against ``state_embed`` and
            extend the candidate set to (B, m+1, ...).
        generator: optional torch.Generator for reproducible softmax
            sampling.

    Returns:
        ``PickResult``: ``action`` (B, H, A), ``chosen_q`` (B,),
        ``chose_dataset`` (B,) bool, ``info`` dict.
    """
    B, m, H, A = optimized_actions.shape
    if state_embed.shape[0] != B:
        raise ValueError(
            f"state_embed batch ({state_embed.shape[0]}) != "
            f"optimized_actions batch ({B})"
        )

    # Optionally extend the candidate set with the dataset action.
    if dataset_action is not None:
        if dataset_action.shape != (B, H, A):
            raise ValueError(
                f"dataset_action shape {tuple(dataset_action.shape)} "
                f"!= expected ({B}, {H}, {A})"
            )
        with torch.no_grad():
            q_data = q_min_from_state_embed(
                critic, state_embed, dataset_action,
            )  # (B,)
        all_actions = torch.cat(
            [optimized_actions, dataset_action.unsqueeze(1)], dim=1,
        )  # (B, m+1, H, A)
        all_q = torch.cat([optimized_q, q_data.unsqueeze(1)], dim=1)
        dataset_idx = m  # index of the appended dataset action
    else:
        all_actions = optimized_actions
        all_q = optimized_q
        dataset_idx = None

    M = all_actions.shape[1]

    if mode == "argmax":
        chosen = all_q.argmax(dim=1)  # (B,)
    elif mode == "softmax":
        if softmax_temperature <= 0:
            raise ValueError(
                f"softmax_temperature must be > 0, got {softmax_temperature}"
            )
        logits = all_q / float(softmax_temperature)
        probs = torch.softmax(logits, dim=1)
        # ``torch.multinomial`` doesn't accept a generator on every
        # backend; fall back to a manual gumbel-max if generator is
        # provided to keep determinism.
        if generator is None:
            chosen = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            gumbel = -torch.log(
                -torch.log(
                    torch.rand(
                        probs.shape, device=probs.device, generator=generator,
                    ).clamp_min(1e-12)
                ).clamp_min(1e-12)
            )
            chosen = (logits + gumbel).argmax(dim=1)
    else:
        raise ValueError(
            f"unknown pick mode {mode!r}; expected 'argmax' or 'softmax'"
        )

    idx_exp = chosen.view(B, 1, 1, 1).expand(B, 1, H, A)
    action = all_actions.gather(dim=1, index=idx_exp).squeeze(1)
    chosen_q = all_q.gather(dim=1, index=chosen.view(B, 1)).squeeze(1)

    if dataset_idx is None:
        chose_dataset = torch.zeros(B, dtype=torch.bool, device=action.device)
    else:
        chose_dataset = chosen == dataset_idx

    info: Dict[str, Any] = {
        "mode": mode,
        "softmax_temperature": float(softmax_temperature),
        "n_candidates": int(M),
        "include_dataset_action": dataset_action is not None,
        "frac_dataset_action_chosen": float(chose_dataset.float().mean().item()),
        "q_chosen_mean": float(chosen_q.mean().item()),
        "q_chosen_std": float(chosen_q.std().item()) if B > 1 else 0.0,
        "q_candidates_std_mean": float(all_q.std(dim=1).mean().item())
            if M > 1 else 0.0,
    }
    return PickResult(
        action=action,
        chosen_q=chosen_q,
        chose_dataset=chose_dataset,
        info=info,
    )


# ---------------------------------------------------------------------------
# End-to-end convenience: propose + global + local + pick.
# ---------------------------------------------------------------------------


def optimize_actions(
    *,
    proposer,
    critic,
    obs_dict: Dict[str, torch.Tensor],
    K: int,
    m: int,
    T: int,
    alpha: float,
    pick_mode: str = "argmax",
    softmax_temperature: float = 1.0,
    dataset_action: Optional[torch.Tensor] = None,
    proposer_inference_steps: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    state_embed_override: Optional[torch.Tensor] = None,
    global_cond_override: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Run the full PA-RL action optimization pipeline.

    Sample K -> rerank top-m -> local Q-grad ascent -> pick (argmax /
    softmax / dataset-augmented). The frozen ``critic`` and the
    ``proposer`` (typically a snapshot of the diffusion policy, e.g. the
    EMA model) are used in eval-only mode; this function never touches
    their parameters.

    Args:
        ... (see other args)
        state_embed_override: optional precomputed ``(B, D)`` state
            embedding produced by the SAME ``critic`` for the SAME
            ``obs_dict``. When provided, we skip
            ``critic.state_embed(obs_dict)`` entirely; intended for
            workflows that pre-compute and cache state embeddings
            once for all dataset frames (the IQL critic's encoder is
            frozen, so its output is invariant across relabel
            cycles). Must be on the same device as the candidate
            actions; dtype is preserved (typically fp16/bf16 if the
            cache was built under autocast).

    Returns:
        ``(action_opt, info)`` where
            action_opt: (B, H, A) UNNORMALIZED chosen action chunk.
            info: dict of stage-wise diagnostics (each prefixed by
                  ``parl/``) suitable for direct logging to wandb.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if m > K:
        raise ValueError(f"m ({m}) > K ({K})")

    B = next(iter(obs_dict.values())).shape[0]

    # Stage I.a: propose K candidates.
    with torch.no_grad():
        candidates = propose_actions(
            proposer, obs_dict, K,
            inference_steps=proposer_inference_steps,
            global_cond_override=global_cond_override,
        )  # (B, K, H, A)

    # Compute state_embed ONCE per state (and reuse it across rerank,
    # local-opt, pick). The critic owns its own normalizer + obs
    # encoder; we just feed it the unnormalized obs dict.
    if state_embed_override is not None:
        # Trust the cache: same critic + same obs -> same embedding.
        # We still validate the leading batch dim to fail loudly on
        # off-by-one indexing bugs in the cache lookup site.
        if state_embed_override.shape[0] != B:
            raise ValueError(
                f"state_embed_override has B={state_embed_override.shape[0]} "
                f"but obs_dict has B={B}"
            )
        state_embed = state_embed_override
        used_cache = True
    else:
        with torch.no_grad():
            state_embed = critic.state_embed(obs_dict, target=False)  # (B, D)
        used_cache = False

    # Stage I.b: global optimization. We need state_embed REPEATED to
    # (B*K, D) so the critic scores each candidate independently.
    state_embed_K = state_embed.repeat_interleave(K, dim=0)  # (B*K, D)
    top_actions, top_q = global_optimize(
        critic, state_embed_K, candidates, m,
    )  # (B, m, H, A), (B, m)

    # Stage I.c: local optimization. Same trick: replicate state_embed
    # to (B*m, D).
    state_embed_m = state_embed.repeat_interleave(m, dim=0)  # (B*m, D)
    H, A = top_actions.shape[2], top_actions.shape[3]
    flat_actions = top_actions.reshape(B * m, H, A)
    flat_actions_opt, flat_q_after, local_info = local_optimize(
        critic, state_embed_m, flat_actions, T=T, alpha=float(alpha),
    )
    actions_opt = flat_actions_opt.view(B, m, H, A)
    q_after = flat_q_after.view(B, m)

    # Stage II: pick.
    pick = pick_optimized(
        critic, state_embed, actions_opt, q_after,
        mode=pick_mode,
        softmax_temperature=softmax_temperature,
        dataset_action=dataset_action,
        generator=generator,
    )

    # Aggregate diagnostics under a parl/ prefix for wandb.
    info: Dict[str, Any] = {
        "parl/K": int(K),
        "parl/m": int(m),
        "parl/T": int(T),
        "parl/alpha": float(alpha),
        "parl/pick_mode": pick_mode,
        "parl/q_global_top1_mean": float(top_q[:, 0].mean().item()),
        "parl/q_global_topm_mean": float(top_q.mean().item()),
        "parl/q_global_spread_mean": float(
            (top_q.max(dim=1).values - top_q.min(dim=1).values).mean().item()
        ) if m > 1 else 0.0,
        "parl/q_before_local_mean": local_info["q_before_mean"],
        "parl/q_after_local_mean": local_info["q_after_mean"],
        "parl/q_local_gain_mean": local_info["q_gain_mean"],
        "parl/frac_local_reverted": local_info["frac_local_reverted"],
        "parl/action_l2_change_mean": local_info["action_l2_change_mean"],
        "parl/n_local_steps_mean": local_info.get("n_steps_mean", 0.0),
        "parl/q_chosen_mean": pick.info["q_chosen_mean"],
        "parl/q_chosen_std": pick.info["q_chosen_std"],
        "parl/q_candidates_std_mean": pick.info["q_candidates_std_mean"],
        "parl/frac_dataset_action_chosen": pick.info["frac_dataset_action_chosen"],
        "parl/include_dataset_action": pick.info["include_dataset_action"],
        "parl/B": int(B),
        "parl/state_embed_cache_hit": float(used_cache),
        "parl/global_cond_cache_hit": float(global_cond_override is not None),
    }
    return pick.action, info


__all__ = [
    "q_min_from_state_embed",
    "propose_actions",
    "global_optimize",
    "local_optimize",
    "pick_optimized",
    "PickResult",
    "optimize_actions",
]
