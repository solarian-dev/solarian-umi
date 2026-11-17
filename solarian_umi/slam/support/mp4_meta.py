"""Camera serial for a GoPro or Lumi collector MP4."""

import av


def camera_serial(mp4_path) -> str:
    """The serial is the GPMF CASN in the gpmd track.

    GoPro and Lumi both write it there. The pipeline uses it to pair a demo
    with that camera's gripper calibration.
    """
    serial = _gpmf_camera_serial(mp4_path)
    if serial:
        return serial
    raise KeyError(f"{mp4_path} has no GPMF CASN camera serial")


def _gpmf_camera_serial(mp4_path) -> str | None:
    with av.open(str(mp4_path)) as container:
        streams = [
            s for s in container.streams
            if s.metadata.get("handler_name") == "GoPro MET"
        ]
        if not streams:
            streams = [s for s in container.streams if s.type == "data"]
        if not streams:
            return None
        blob = b""
        for packet in container.demux(streams):
            if packet.size:
                blob += bytes(packet)
            if b"CASN" in blob or len(blob) > 1_000_000:
                break
    return _casn_text(blob)


def _casn_text(blob: bytes) -> str | None:
    idx = blob.find(b"CASN")
    if idx < 0 or idx + 8 > len(blob):
        return None
    type_char = blob[idx + 4]
    size = blob[idx + 5]
    repeat = int.from_bytes(blob[idx + 6:idx + 8], "big")
    end = idx + 8 + size * repeat
    if type_char != ord("c") or size == 0 or repeat == 0 or end > len(blob):
        return None
    text = blob[idx + 8:end].split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()
    return text or None
