# Adapted from Universal Manipulation Interface.
# Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab.
# https://github.com/real-stanford/universal_manipulation_interface
# Licensed under the MIT License. The full text is LICENSE.umi
# beside this package.

from typing import Optional, Union
from fractions import Fraction
import datetime
import os
import av


def timecode_to_seconds(
        timecode: str, frame_rate: Union[int, float, Fraction]
        ) -> Union[float, Fraction]:
    """
    Convert non-skip frame timecode into seconds since midnight
    """
    # calculate whole frame rate
    # 29.97 -> 30, 59.94 -> 60
    int_frame_rate = round(frame_rate)

    # parse timecode string
    h, m, s, f = [int(x) for x in timecode.split(':')]

    # calculate frames assuming whole frame rate (i.e. non-drop frame)
    frames = (3600 * h + 60 * m + s) * int_frame_rate + f

    # convert to seconds
    seconds = frames / frame_rate
    return seconds


def stream_get_start_datetime(
        stream: av.stream.Stream,
        recorded_at: Optional[datetime.datetime] = None,
        ) -> datetime.datetime:
    """
    Combines creation time and timecode to get high-precision
    time for the first frame of a video.
    """
    # read metadata
    frame_rate = stream.average_rate
    tc = stream.metadata['timecode']
    creation_time = stream.metadata.get('creation_time')
    
    # get time within the day
    seconds_since_midnight = float(timecode_to_seconds(timecode=tc, frame_rate=frame_rate))
    delta = datetime.timedelta(seconds=seconds_since_midnight)

    # GoPro stamps creation_time. Lumi files muxed without it still have a
    # timecode; the file mtime is the seal time, so its UTC date is the
    # recording date unless the take crosses UTC midnight.
    if not creation_time:
        if recorded_at is None:
            raise KeyError('creation_time')
        recorded_utc = recorded_at.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        day = recorded_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        start_datetime = day + delta
        if start_datetime > recorded_utc + datetime.timedelta(minutes=5):
            start_datetime -= datetime.timedelta(days=1)
        return start_datetime

    # get dates
    create_datetime = datetime.datetime.strptime(creation_time, r"%Y-%m-%dT%H:%M:%S.%fZ")
    create_datetime = create_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
    start_datetime = create_datetime + delta
    return start_datetime


def mp4_get_start_datetime(mp4_path: str) -> datetime.datetime:
    recorded_at = datetime.datetime.fromtimestamp(
        os.path.getmtime(mp4_path), datetime.timezone.utc)
    with av.open(mp4_path) as container:
        stream = container.streams.video[0]
        return stream_get_start_datetime(stream=stream, recorded_at=recorded_at)
