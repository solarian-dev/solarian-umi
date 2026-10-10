"""feature_aggregation=box_pool: patch tokens under the target box condition the policy."""

import pytest
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from solarian_umi_skill.diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from solarian_umi_skill.diffusion_policy.model.vision.timm_obs_encoder import (
    TimmObsEncoder,
    pool_box_tokens,
    widen_cond_for_box_pool,
)
from solarian_umi_skill.diffusion_policy.policy.diffusion_unet_timm_policy import (
    DiffusionUnetTimmPolicy,
)


def _shape_meta(with_bbox: bool = True) -> dict:
    obs = {
        "camera0_rgb": {"shape": [3, 224, 224], "horizon": 1, "type": "rgb"},
        "robot0_gripper_width": {"shape": [1], "horizon": 2, "type": "low_dim"},
    }
    if with_bbox:
        obs["bbox"] = {
            "shape": [4], "horizon": 1, "type": "low_dim",
            "bbox_mode": "mask_channel", "target_rgb_key": "camera0_rgb",
        }
    return {"obs": obs, "action": {"shape": [10], "horizon": 16}}


def _encoder(feature_aggregation, with_bbox: bool = True) -> TimmObsEncoder:
    return TimmObsEncoder(
        shape_meta=_shape_meta(with_bbox),
        model_name="vit_tiny_patch16_224",
        pretrained=False,
        frozen=False,
        global_pool="",
        transforms=None,
        feature_aggregation=feature_aggregation,
    ).eval()


def _obs(bbox) -> dict:
    return {
        "camera0_rgb": torch.rand(2, 1, 3, 224, 224),
        "robot0_gripper_width": torch.rand(2, 2, 1),
        "bbox": torch.tensor(bbox, dtype=torch.float32).view(2, 1, 4),
    }


def test_box_mean_averages_only_the_patches_under_the_box():
    grid = (2, 2)
    tokens = torch.zeros(1, 1 + 4, 3)
    tokens[0, 0] = 9.0
    tokens[0, 1:, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., :, 2:] = 1.0

    out = pool_box_tokens(tokens, mask, grid, num_prefix_tokens=1)

    assert out.shape == (1, 2 * 3 + 5)
    assert torch.all(out[0, :3] == 9.0)
    assert out[0, 3] == pytest.approx(3.0)
    assert out[0, 6:].tolist() == pytest.approx([0.0, -1.0, 1.0, 1.0, 1.0])


def test_empty_box_zeroes_everything_after_cls():
    tokens = torch.randn(1, 1 + 4, 3)

    out = pool_box_tokens(tokens, torch.zeros(1, 1, 4, 4), (2, 2), num_prefix_tokens=1)

    assert torch.equal(out[0, :3], tokens[0, 0])
    assert torch.count_nonzero(out[0, 3:]) == 0


def test_box_pool_widens_the_conditioning_and_follows_the_box():
    torch.manual_seed(0)
    cls_enc = _encoder(None)
    box_enc = _encoder("box_pool")
    width = cls_enc.key_model_map["camera0_rgb"].num_features

    assert cls_enc.output_shape() == (1, width + 2)
    assert box_enc.output_shape() == (1, 2 * width + 5 + 2)

    obs = _obs([[16, 16, 64, 64], [160, 160, 208, 208]])
    with torch.no_grad():
        feat = box_enc(obs)
    corners = feat[:, 2 * width:2 * width + 5]
    assert corners[0].tolist() == pytest.approx([16 / 112 - 1, 16 / 112 - 1, 64 / 112 - 1, 64 / 112 - 1, 1.0])
    assert corners[1, 0] > 0 and corners[1, 4] == 1.0


def _policy(feature_aggregation) -> DiffusionUnetTimmPolicy:
    torch.manual_seed(0)
    policy = DiffusionUnetTimmPolicy(
        shape_meta=_shape_meta(),
        noise_scheduler=DDPMScheduler(num_train_timesteps=4),
        obs_encoder=_encoder(feature_aggregation),
        diffusion_step_embed_dim=16,
        down_dims=(16, 32),
        n_groups=4,
    ).eval()
    normalizer = LinearNormalizer()
    for key in ("camera0_rgb", "robot0_gripper_width", "bbox", "action"):
        normalizer[key] = SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)
    return policy


def test_widened_cls_checkpoint_predicts_the_same_actions():
    cls_policy = _policy(None)
    box_policy = _policy("box_pool")
    state = cls_policy.state_dict()

    widened = widen_cond_for_box_pool(state, box_policy.obs_encoder, box_policy.state_dict())
    box_policy.load_state_dict(state)

    # Two residual blocks per down level and per up level past the first, two mid.
    assert widened == 8
    obs = _obs([[16, 16, 64, 64], [160, 160, 208, 208]])
    with torch.no_grad():
        torch.manual_seed(1)
        expected = cls_policy.predict_action(obs)["action"]
        torch.manual_seed(1)
        actual = box_policy.predict_action(obs)["action"]
    assert torch.allclose(actual, expected, atol=1e-5)
    assert widen_cond_for_box_pool(state, box_policy.obs_encoder, box_policy.state_dict()) == 0


def test_box_pool_requires_a_mask_channel_box():
    with pytest.raises(RuntimeError, match="mask_channel"):
        _encoder("box_pool", with_bbox=False)
