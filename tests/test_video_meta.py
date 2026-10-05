"""Start time and camera serial for GoPro and Lumi MP4s."""

import datetime

from solarian_umi_skill.slam.support.mp4_meta import _casn_text
from solarian_umi_skill.slam.support.timecode_util import stream_get_start_datetime


class _Stream:
    def __init__(self, metadata, rate=30):
        self.metadata = metadata
        self.average_rate = rate


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


def test_gpmf_casn_is_the_camera_serial():
    blob = b"xxxxCASN" + bytes((ord("c"), 12)) + (1).to_bytes(2, "big") + b"LUMI07FC9785"
    assert _casn_text(blob) == "LUMI07FC9785"
