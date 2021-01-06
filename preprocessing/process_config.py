import yaml
from pathlib import Path
from addict import Dict
import numpy as np
import re
import os
from PIL import Image
from anipose_BA import CameraGroup, Camera


def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        try:
            num, denom = frac_str.split("/")
            if denom == "pi":
                denom = np.pi
        except ValueError:
            return None
        try:
            leading, num = num.split(" ")
        except ValueError:
            return float(num) / float(denom)
        if float(leading) < 0:
            sign_mult = -1
        else:
            sign_mult = 1

        return float(leading) + sign_mult * (float(num) / float(denom))


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanumeric_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]

    if ".DS_Store" in l:
        l.remove(".DS_Store")

    return sorted(l, key=alphanumeric_key)


def read_yaml(path_to_yaml, frame_to_skip=None):
    assert isinstance(path_to_yaml, str)
    path_to_yaml = Path(path_to_yaml)
    with open(path_to_yaml, "r") as f:
        config = Dict(yaml.safe_load(f))

    # Get points
    points_2d_joints = np.load(config.path_to_points)
    if config.path_to_likelihoods:
        likelihoods = np.load(config.path_to_likelihoods)
    else:
        likelihoods = np.empty((points_2d_joints.shape[0], points_2d_joints.shape[1], points_2d_joints.shape[2]))
        likelihoods[:] = np.nan
        
    num_cams = points_2d_joints.shape[0]
    num_frames = points_2d_joints.shape[1]
    num_bodyparts = points_2d_joints.shape[2]

    # Get image paths and dimensions
    path_to_views = Path(config.path_to_images).resolve()
    views_dirs = sorted_nicely(os.listdir(path_to_views))
    assert num_cams == len(
        views_dirs
    ), "Mismatch between number of views in points, and number of views in frames path"

    frame_paths = []
    image_heights = []
    image_widths = []

    for view in views_dirs:
        view_path = Path(path_to_views / view)
        view_frames = sorted_nicely(os.listdir(view_path))
        view_frames = [view_path / i for i in view_frames]
        if num_frames != len(view_frames):
            if num_frames - 1 == len(view_frames):

                assert frame_to_skip is not None, """Mismatch between number of frames in points and number of frames;  
                specify which frame to skip (e.g --skip_frame -1 or --skip_frame 0)"""
                
                if frame_to_skip == 0:
                    points_2d_joints = points_2d_joints[:, 1:, :, :]
                    likelihoods = likelihoods[:, 1:, :]
                elif frame_to_skip == -1:
                    points_2d_joints = points_2d_joints[:, :-1, :, :]
                    likelihoods = likelihoods[:, :-1, :]

                else:
                    points_2d_joints = points_2d_joints[:, :frame_to_skip, frame_to_skip:, :, :]
                    likelihoods = likelihoods[:, :frame_to_skip, frame_to_skip:, :]

                num_frames -= 1
            else:
                raise ValueError(
                    "Mismatch between number of frames in points and number of frames"
                )

        frame_paths.append(view_frames)

        # Select a random frame and read the image to get dimensions
        idx = np.random.choice(len(view_frames))
        img = Image.open(view_frames[idx])
        width, height = img.size
        image_heights.append(height)
        image_widths.append(width)

    # Get focal lengths
    focal_lengths = []
    for i in range(num_cams):
        focal_length = (
            config.intrinsics.focal_length.focal_length_mm
            * ((image_heights[i] ** 2 + image_widths[i] ** 2) ** (1 / 2))
        ) / config.intrinsics.focal_length.sensor_size

        focal_lengths.append(focal_length)

    # Get cam group based on specifications
    cameras = []
    translations = config.extrinsics.translation
    rotations = config.extrinsics.rotation

    for i in range(num_cams):
        if f"cam{i + 1}" not in translations:
            tvec = [0, 0, 0]
        else:
            tvec = translations[f"cam{i + 1}"]

        if f"cam{i + 1}" not in rotations:
            rvec = [0, 0, 0]
        else:
            rvec = [
                convert_to_float(rotations[f"cam{i + 1}"][0]) * np.pi,
                convert_to_float(rotations[f"cam{i + 1}"][1]) * np.pi,
                convert_to_float(rotations[f"cam{i + 1}"][2]) * np.pi,
            ]

        cam = Camera(rvec=rvec, tvec=tvec)

        cam.set_focal_length(focal_lengths[i])

        cam_mat = cam.get_camera_matrix()
        cam_mat[0, 2] = image_widths[i] // 2
        cam_mat[1, 2] = image_heights[i] // 2
        cam.set_camera_matrix(cam_mat)
        cameras.append(cam)

    cam_group = CameraGroup(cameras=cameras)

    return {
        "cam_group": cam_group,
        "config": config,
        "points_2d_joints": points_2d_joints,
        "likelihoods": likelihoods,
        "frame_paths": frame_paths,
        "num_cams": num_cams,
        "num_frames": num_frames,
        "num_bodyparts": num_bodyparts,
        "image_heights": image_heights,
        "image_widths": image_widths,
        "focal_length": focal_lengths,
    }
