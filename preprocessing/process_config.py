import yaml
from pathlib import Path
from addict import Dict
import numpy as np
import re
import os
import sys
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.resolve()))
import cv2
from anipose_BA import CameraGroup, Camera
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Process config")
    # dataset
    parser.add_argument("--path_to_yaml", type=str, required=True)
    parser.add_argument("--csv_type", type=str, required=True)

    return parser.parse_args()


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


def process_dlc_csv(path_to_csv, bp_to_keep=None, lh_thresh=0.9):
    points_df = pd.read_csv(path_to_csv, header=[0, 1, 2], chunksize=10000)
    points_df = pd.concat([i for i in tqdm(points_df)], ignore_index=True)

    points_and_confs = points_df.values[:, 1:]
    num_bodyparts = int(((points_df.columns.shape[0]) - 1) / 3)

    points_arr = np.empty((points_and_confs.shape[0], num_bodyparts, 2))
    confs = np.empty((points_and_confs.shape[0], num_bodyparts))
    for i in range(num_bodyparts):
        x = points_and_confs[:, 3 * i]
        y = points_and_confs[:, 3 * i + 1]
        lh = points_and_confs[:, 3 * i + 2]
        points_arr[:, i, 0] = x
        points_arr[:, i, 1] = y
        confs[:, i] = lh

    points_arr[confs < lh_thresh] = np.nan
    return points_arr, confs


def read_yaml(path_to_yaml, csv_type):
    assert isinstance(path_to_yaml, str)
    path_to_yaml = Path(path_to_yaml)
    with open(path_to_yaml, "r") as f:
        config = Dict(yaml.safe_load(f))

    assert csv_type in ["dlc", "sawtell"]

    path_to_csvs = config.path_to_csv
    path_to_videos = config.path_to_videos

    point_sizes = config.point_sizes

    if config.mirrored:
        assert len(path_to_csvs) == 1
        assert config.image_limits

    assert len(path_to_csvs) == len(path_to_videos)

    if csv_type == "dlc":
        points_2d_joints = []
        likelihoods = []
        for csv_file_path in path_to_csvs:
            curr_points, curr_likelihoods = process_dlc_csv(csv_file_path)
            points_2d_joints.append(curr_points)
            likelihoods.append(curr_likelihoods)
        points_2d_joints = np.asarray(points_2d_joints)
        likelihoods = np.asarray(likelihoods)

    elif csv_type == "sawtell":
        from preprocessing.preprocess_Sawtell import get_data

        assert config.mirrored

        img_settings = config.image_limits
        points_2d_joints, likelihoods, img_settings = get_data(
            img_settings=img_settings,
            dlc_file=path_to_csvs[0],
            save_arrays=False,
            chunksize=10000,
            bp_to_keep=config.bp_names
        )
    else:
        raise ValueError(f"csv_type {csv_type} is invalid.")

    num_cams = points_2d_joints.shape[0]
    num_frames = points_2d_joints.shape[1]
    num_bodyparts = points_2d_joints.shape[2]

    image_heights = []
    image_widths = []

    if config.mirrored:
        vid = path_to_videos[0]
        cap = cv2.VideoCapture(str(vid))
        success, image = cap.read()
        total_height, total_width, layers = image.shape
        for i in range(num_cams):
            sub_height, sub_width, layers = image[
                config.image_limits["height_lims"][i][0]: config.image_limits["height_lims"][i][1],
                config.image_limits["width_lims"][i][0]: config.image_limits["width_lims"][i][1],
                :,
            ].shape
            image_heights.append(sub_height)
            image_widths.append(sub_width)
    else:
        for vid in path_to_videos:
            cap = cv2.VideoCapture(str(vid))
            success, image = cap.read()
            height, width, layers = image.shape
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
        "num_cams": num_cams,
        "num_frames": num_frames,
        "num_bodyparts": num_bodyparts,
        "image_heights": image_heights,
        "image_widths": image_widths,
        "focal_length": focal_lengths,
        "video_paths": path_to_videos,
        "point_sizes": point_sizes,
        "img_settings": config.image_limits,
        "image_heights": image_heights,
        "image_widths": image_widths
    }


if __name__ == "__main__":
    args = get_args()
    experiment_data = read_yaml(args.path_to_yaml, args.csv_type)
