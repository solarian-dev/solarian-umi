import cv2
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from solarian.core.optical_remap import (
    GOPRO_OUT_SIZE,
    Aligner,
    identity_aligner,
    load_shipped_gopro_intrinsics,
)

from solarian_umi.commands.calibrate_umi_camera import (
    _resolve_gold,
    _solve_view_rotation,
    _upsert_yaml_key,
)
from solarian_umi.umi.compare_policy_image import DEFAULT_GOLD
from solarian_umi.umi.gripper_align import RegionPolygons


def test_shipped_gold_is_the_224_policy_crop():
    gold = cv2.imread(str(DEFAULT_GOLD), cv2.IMREAD_COLOR)
    assert gold is not None
    assert gold.shape[:2] == (224, 224)


def test_resolve_gold_defaults_to_shipped_and_honors_override(tmp_path):
    default = _resolve_gold(OmegaConf.create({}), OmegaConf.create({}))
    assert default == DEFAULT_GOLD

    custom = tmp_path / "custom_gold.png"
    cv2.imwrite(str(custom), np.zeros((224, 224, 3), dtype=np.uint8))
    override = _resolve_gold(OmegaConf.create({"gold": str(custom)}), OmegaConf.create({}))
    assert override == custom


def test_view_rotation_recovers_a_camera_tilted_on_its_mount():
    """Outlines seen through a view pitched 4.6 deg come back onto gold."""
    intr = load_shipped_gopro_intrinsics()
    aligner = identity_aligner(intr, intr)
    tilt = Rotation.from_rotvec(np.radians([-4.6, -0.8, 0.1])).as_matrix()
    tilted = Aligner(intr, intr, tilt.T, out_size=GOPRO_OUT_SIZE)
    scale, x0 = 1080 / 224, (1920 - 1080) / 2

    def seen_tilted(poly):
        px = np.array([[x0 + x * scale, y * scale] for x, y in poly])
        out = tilted.forward_points(px)
        return [((u - x0) / scale, v / scale) for u, v in out]

    gold = RegionPolygons(
        orange=[[(65, 158), (93, 145), (74, 177)], [(130, 146), (148, 178), (157, 158)]],
        white=[[(42, 169), (60, 155), (72, 175), (71, 182)]],
    )
    live = RegionPolygons(
        orange=[seen_tilted(p) for p in gold.orange],
        white=[seen_tilted(p) for p in gold.white],
    )

    rot, before, after, move = _solve_view_rotation(gold, live, aligner, (224, 224))
    assert before > 4.0 and after < 0.5
    assert np.degrees(np.linalg.norm(Rotation.from_matrix(rot @ tilt.T).as_rotvec())) < 0.3
    assert np.allclose(move(live.orange[0]), gold.orange[0], atol=1.0)


def test_yaml_patch_sets_optical_remap_and_16_9_resize():
    text = (
        "sensors:\n"
        "  camera:\n"
        "    width: 2028\n"
        "    height: 1520\n"
        "    resize_width: 480\n"
        "    resize_height: 360\n"
        "    type: picamera\n"
    )
    text = _upsert_yaml_key(text, "optical_remap", "conf/calibrations/cam_gopro_remap.json")
    text = _upsert_yaml_key(text, "resize_width", "640")
    text = _upsert_yaml_key(text, "resize_height", "360")
    assert "optical_remap: conf/calibrations/cam_gopro_remap.json" in text
    assert "resize_width: 640" in text
    assert "resize_height: 360" in text
    assert "resize_width: 480" not in text
