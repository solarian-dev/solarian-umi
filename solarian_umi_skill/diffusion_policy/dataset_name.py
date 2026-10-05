# Adapted from Universal Manipulation Interface.
# Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab.
# https://github.com/real-stanford/universal_manipulation_interface
# Licensed under the MIT License. The full text is LICENSE.umi
# beside this package.

import os
import re


def parse_dataset_field(dataset_path: str, field: str, default=None):
    """Parse a value from the dataset filename's hyphen-separated tokens.

    Expected filename format (extension `.zarr.zip` or similar):
        <data_task>-<mirror>-...-real_<N>-...-sim_<N>-...-(bbox|no_bbox).zarr.zip

    Recognized fields: data_task, mirror, real_data_size, sim_data_size, bbox.
    Returns `default` if the field cannot be inferred.
    """
    name = os.path.basename(str(dataset_path)).split('.')[0]
    tokens = [t for t in name.split('-') if t]

    if field == "data_task":
        return tokens[0] if tokens else default
    if field == "mirror":
        valid = {"black_mirror", "mirror", "swap_mirror", "no_mirror"}
        for t in tokens:
            if t in valid:
                return t
        return default
    if field == "real_data_size":
        for t in tokens:
            m = re.fullmatch(r"real_(\d+)", t)
            if m:
                return int(m.group(1))
        return default
    if field == "sim_data_size":
        for t in tokens:
            m = re.fullmatch(r"sim_(\d+)", t)
            if m:
                return int(m.group(1))
        return default
    if field == "bbox":
        if "bbox" in tokens:
            return "bbox"
        if "no_bbox" in tokens:
            return "no_bbox"
        return default
    raise ValueError(f"Unknown dataset field: {field!r}")
