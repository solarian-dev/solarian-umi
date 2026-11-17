"""Start time and camera serial for GoPro and Lumi MP4s."""

import datetime
from fractions import Fraction

from solarian_umi.slam.support.mp4_meta import _casn_text
from solarian_umi.slam.support.timecode_util import stream_get_start_datetime


class _Stream:
    def __init__(self, metadata, rate=30, average_rate=None):
        self.metadata = metadata
        self.base_rate = rate
        self.average_rate = rate if average_rate is None else average_rate


def test_creation_time_sets_the_date():
    stream = _Stream({
        "timecode": "18:02:38:01",
        "creation_time": "2024-01-05T01:02:03.000000Z",
    })
    start = stream_get_start_datetime(stream)
    assert start == datetime.datetime(2024, 1, 5, 18, 2, 38, 33333)


def test_missing_creation_time_uses_the_seal_date():
    stream = _Stream({"timecode": "18:02:38:01"})
    recorded = datetime.datetime(2026, 9, 30, 18, 2, 49, tzinfo=datetime.timezone.utc)
    start = stream_get_start_datetime(stream, recorded_at=recorded)
    assert start == datetime.datetime(2026, 9, 30, 18, 2, 38, 33333)


def test_a_take_that_crosses_utc_midnight_keeps_the_start_day():
    stream = _Stream({"timecode": "23:59:00:00"})
    recorded = datetime.datetime(2026, 10, 1, 0, 1, tzinfo=datetime.timezone.utc)
    start = stream_get_start_datetime(stream, recorded_at=recorded)
    assert start == datetime.datetime(2026, 9, 30, 23, 59, 0)


def test_dropped_frames_do_not_move_the_start():
    stream = _Stream({"timecode": "18:08:42:04"}, average_rate=Fraction(6240, 217))
    recorded = datetime.datetime(2026, 10, 14, 18, 8, 56, tzinfo=datetime.timezone.utc)
    start = stream_get_start_datetime(stream, recorded_at=recorded)
    assert start == datetime.datetime(2026, 10, 14, 18, 8, 42, 133333)


def test_gpmf_casn_is_the_camera_serial():
    blob = b"xxxxCASN" + bytes((ord("c"), 12)) + (1).to_bytes(2, "big") + b"LUMI07FC9785"
    assert _casn_text(blob) == "LUMI07FC9785"
