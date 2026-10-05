import shutil
from fractions import Fraction

import av
import numpy as np
import pytest

from solarian_umi_skill.slam.fisheye_model import Fisheye, build_remap
from solarian_umi_skill.slam.remap_video import write_remapped

SMALL = Fisheye(fx=58.2, fy=58.2, cx=120.0, cy=90.0, k=(-0.0088, -0.0016, -0.00084, -0.000049), width=240, height=180)


def _write_native(path, levels):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = SMALL.width, SMALL.height, "yuv420p"
        stream.options = {"crf": "0"}
        for k, level in enumerate(levels):
            frame = av.VideoFrame.from_ndarray(np.full((SMALL.height, SMALL.width, 3), level, np.uint8), format="rgb24")
            frame.pts, frame.time_base = k, Fraction(1, 30)
            container.mux(stream.encode(frame))
        container.mux(stream.encode())


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_dropped_frame_repeats_the_one_before_it(tmp_path):
    native, out = tmp_path / "native.mp4", tmp_path / "out.mp4"
    levels = [40, 90, 140, 190]
    _write_native(native, levels)
    frame_t = np.array([0.0, 1 / 30, 3 / 30, 4 / 30])  # the frame at 2/30 was dropped

    write_remapped(native, out, build_remap(SMALL, SMALL), frame_t)

    with av.open(str(out)) as container:
        stream = container.streams.video[0]
        assert stream.average_rate == 30
        centers = [int(f.to_ndarray(format="rgb24")[90, 120].mean()) for f in container.decode(stream)]
    assert centers == pytest.approx([40, 90, 90, 140, 190], abs=4)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_sidecar_frame_count_must_match_the_video(tmp_path):
    native = tmp_path / "native.mp4"
    _write_native(native, [40, 90, 140])

    with pytest.raises(SystemExit, match="3 frames but its sidecar times 2"):
        write_remapped(native, tmp_path / "out.mp4", build_remap(SMALL, SMALL), np.array([0.0, 1 / 30]))
    assert not (tmp_path / "out.mp4").exists()
