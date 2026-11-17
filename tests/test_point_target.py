"""Point targets, and several targets on one image: one ViT channel each."""

import pytest
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from solarian_umi.diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from solarian_umi.diffusion_policy.model.vision.timm_obs_encoder import (
    TimmObsEncoder,
    rasterize_bbox_to_mask,
    rasterize_point_to_mask,
    widen_cond_for_box_pool,
    widen_patch_embed,
)
from solarian_umi.diffusion_policy.policy.diffusion_unet_timm_policy import (
    DiffusionUnetTimmPolicy,
)
from solarian_umi.targets import target_keys

BOX = {"shape": [4], "horizon": 1, "type": "low_dim", "bbox_mode": "mask_channel",
       "target_rgb_key": "camera0_rgb"}
POINT = {"shape": [2], "horizon": 1, "type": "low_dim", "point_mode": "mask_channel",
         "point_box_px": 16, "target_rgb_key": "camera0_rgb"}


def _meta(**targets) -> dict:
    obs = {
        "camera0_rgb": {"shape": [3, 224, 224], "horizon": 1, "type": "rgb"},
        "robot0_gripper_width": {"shape": [1], "horizon": 2, "type": "low_dim"},
        **targets,
    }
    return {"obs": obs, "action": {"shape": [10], "horizon": 16}}


def _encoder(meta, feature_aggregation=None) -> TimmObsEncoder:
    return TimmObsEncoder(
        shape_meta=meta, model_name="vit_tiny_patch16_224", pretrained=False, frozen=False,
        global_pool="", transforms=None, feature_aggregation=feature_aggregation,
    ).eval()


def _obs(**targets) -> dict:
    torch.manual_seed(3)
    return {"camera0_rgb": torch.rand(2, 1, 3, 224, 224), "robot0_gripper_width": torch.rand(2, 2, 1),
            **{k: torch.tensor(v, dtype=torch.float32).view(2, 1, -1) for k, v in targets.items()}}


def test_the_target_kinds_are_read_off_shape_meta():
    assert target_keys(_meta(bbox=BOX, place=POINT)["obs"]) == {"bbox": "box", "place": "point"}
    assert target_keys(_meta()["obs"]) == {}


def test_a_point_is_the_square_centred_on_it_and_a_negative_one_is_nothing():
    mask = rasterize_point_to_mask(torch.tensor([[[20.0, 30.0]], [[-1.0, -1.0]]]), H=64, W=64, side=8)
    assert mask.shape == (2, 1, 1, 64, 64)
    assert torch.equal(mask[0], rasterize_bbox_to_mask(torch.tensor([[16.0, 26.0, 24.0, 34.0]]), H=64, W=64))
    assert mask[1].sum() == 0


def test_a_point_conditions_exactly_like_its_square_as_a_box():
    torch.manual_seed(0)
    box_enc = _encoder(_meta(bbox=BOX), "box_pool")
    with torch.no_grad():
        box_enc.key_model_map["camera0_rgb"].patch_embed.proj.weight[:, 3].normal_()
    point_enc = _encoder(_meta(point=POINT), "box_pool")
    point_enc.load_state_dict(box_enc.state_dict(), strict=True)

    with torch.no_grad():
        from_point = point_enc(_obs(point=[[40, 50], [150, 100]]))
        from_box = box_enc(_obs(bbox=[[32, 42, 48, 58], [142, 92, 158, 108]]))
    torch.testing.assert_close(from_point, from_box)


def test_two_targets_pool_one_block_each():
    enc = _encoder(_meta(bbox=BOX, point=POINT), "box_pool")
    width = enc.key_model_map["camera0_rgb"].num_features
    assert enc.key_model_map["camera0_rgb"].patch_embed.proj.in_channels == 5
    assert enc.output_shape() == (1, width + 2 * (width + 5) + 2)

    with torch.no_grad():
        feat = enc(_obs(bbox=[[16, 16, 64, 64]] * 2, point=[[120, 120], [-1, -1]]))
    point_geometry = feat[:, 2 * width + 5 + width:2 * width + 5 + width + 5]
    assert point_geometry[0].tolist() == pytest.approx([112 / 112 - 1, 112 / 112 - 1, 128 / 112 - 1,
                                                        128 / 112 - 1, 1.0])
    assert point_geometry[1].tolist() == [0.0] * 5, "no point, no geometry"


def _policy(meta, feature_aggregation="box_pool") -> DiffusionUnetTimmPolicy:
    torch.manual_seed(0)
    policy = DiffusionUnetTimmPolicy(
        shape_meta=meta, noise_scheduler=DDPMScheduler(num_train_timesteps=4),
        obs_encoder=_encoder(meta, feature_aggregation), diffusion_step_embed_dim=16,
        down_dims=(16, 32), n_groups=4,
    ).eval()
    _set_identity_normalizer(policy, meta)
    return policy


def _set_identity_normalizer(policy, meta) -> None:
    normalizer = LinearNormalizer()
    for key in (*meta["obs"], "action"):
        normalizer[key] = SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)


def test_a_one_target_checkpoint_widened_to_two_predicts_the_same_actions():
    pick = _policy(_meta(bbox=BOX))
    with torch.no_grad():
        pick.obs_encoder.key_model_map["camera0_rgb"].patch_embed.proj.weight[:, 3].normal_()
    both_meta = _meta(bbox=BOX, point=POINT)
    both = _policy(both_meta)
    state = pick.state_dict()

    assert widen_patch_embed(state, both.state_dict()) == 1
    assert widen_cond_for_box_pool(state, both.obs_encoder, both.state_dict()) == 8
    both.load_state_dict(state)
    # The workspace fits the normalizer to the dataset after loading.
    _set_identity_normalizer(both, both_meta)

    obs = _obs(bbox=[[16, 16, 64, 64], [160, 160, 208, 208]], point=[[100, 100], [30, 40]])
    with torch.no_grad():
        torch.manual_seed(1)
        expected = pick.predict_action({k: v for k, v in obs.items() if k != "point"})["action"]
        torch.manual_seed(1)
        actual = both.predict_action(obs)["action"]
    assert torch.allclose(actual, expected, atol=1e-5), "the new channel and its pooled block start at zero"
    assert widen_patch_embed(state, both.state_dict()) == 0


def test_a_cls_checkpoint_widens_to_two_pooled_targets():
    cls_policy = _policy(_meta(bbox=BOX), feature_aggregation=None)
    both = _policy(_meta(bbox=BOX, point=POINT))
    state = cls_policy.state_dict()
    widen_patch_embed(state, both.state_dict())
    assert widen_cond_for_box_pool(state, both.obs_encoder, both.state_dict()) == 8
    both.load_state_dict(state)


def test_the_place_template_observes_a_pooled_point():
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from solarian.core.ai.system1.trainer_registry import load_train_config

    training = Path(__file__).resolve().parents[1] / "solarian_umi" / "templates" / "training"
    with initialize_config_dir(version_base=None, config_dir=str(training)):
        umi = load_train_config(compose(config_name="umi_place"))[2].umi_config()

    obs = umi["task"]["shape_meta"]["obs"]
    assert target_keys(obs) == {"point": "point"}
    assert obs["point"]["horizon"] == obs["camera0_rgb"]["horizon"]
    assert umi["policy"]["obs_encoder"]["feature_aggregation"] == "box_pool"


def test_a_point_needs_two_coordinates():
    with pytest.raises(RuntimeError, match=r"point_mode=mask_channel requires shape \[2\]"):
        _encoder(_meta(point={**POINT, "shape": [4]}))
