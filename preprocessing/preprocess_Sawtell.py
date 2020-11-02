from __future__ import print_function
import os
import numpy as np
from pathlib import Path
import re
from utils.utils_IO import (
    ordered_arr_3d_to_dict,
    refill_nan_array,
    arr_2d_to_list_of_dicts,
    sorted_alphanumeric,
)


def get_data():
    import h5py

    data_dir = Path("./data/Sawtell-data").resolve()
    filename = data_dir / "tank_dataset_5.h5"
    f = h5py.File(filename, "r")
    print("-------------DATASET INFO--------------")
    print("keys: ", f.keys())
    print("annotated: ", f["annotated"])
    print("annotations: ", f["annotations"])
    print("images: ", f["images"])
    print("skeleton: ", f["skeleton"])
    print("skeleton_names: ", f["skeleton_names"])
    print("----------------------------------------")

    # Read frame limits:
    import commentjson

    img_settings_file = data_dir / "image_settings.json"
    img_settings = commentjson.load(open(str(img_settings_file), "r"))
    num_cameras = len(img_settings["height_lims"])

    # Get points from hd5
    pts_array = np.asarray(f["annotations"])

    # Get number of frames
    num_frames = pts_array.shape[0]

    # Get number of bodyparts
    num_analyzed_body_parts = int(pts_array.shape[1] / num_cameras)

    # Split points into separate views
    multiview_idx_to_name = {}
    multiview_name_to_idx = {}

    # NOTE: the order should match the order in `image_settings.json`
    view_names = ["top", "main", "right"]
    assert len(view_names) == num_cameras


    # NOTE: Empty list keeps all bodyparts
    bp_to_keep = ["head", "mid", "pectoral"]  # ["head", "chin"]

    for view_name in view_names:
        multiview_name_to_idx[view_name] = []

    new_skeleton_names = []
    for idx, name in enumerate(f["skeleton_names"]):
        if len(bp_to_keep) > 0:
            skip_bp = True
            for bp in bp_to_keep:
                if bp == name.decode("UTF-8").split("_")[0]:
                    skip_bp = False
            if skip_bp:
                continue

        new_skeleton_names.append(name)
        for view_name in view_names:
            if view_name in name.decode("UTF-8").split("_")[-1]:
                multiview_idx_to_name[idx] = view_name
                multiview_name_to_idx[view_name].append(idx)

    if len(bp_to_keep) > 0:
        num_analyzed_body_parts = int(len(new_skeleton_names) / num_cameras)


    # (num views, num frames, num points per frame, 2)
    multiview_pts_2d = np.empty(    
        shape=(num_cameras, pts_array.shape[0], num_analyzed_body_parts, 2)
    )
    for i, view_name in enumerate(view_names):
        # Select rows from indices
        view_indices = multiview_name_to_idx[view_name]
        view_points = pts_array[:, view_indices, :]
        view_points[:, :, 0] -= img_settings["width_lims"][i][0]
        view_points[:, :, 1] -= img_settings["height_lims"][i][0]
        multiview_pts_2d[i, :, :, :] = view_points

    # Now, convert to (num views, num points * num frames, 2)
    multiview_pts_2d = np.reshape(
        multiview_pts_2d,
        (
            multiview_pts_2d.shape[0],
            multiview_pts_2d.shape[1] * multiview_pts_2d.shape[2],
            -1,
        ),
    )

    assert multiview_pts_2d.shape[-1] == 2

    num_points_all = multiview_pts_2d.shape[1]

    # pts_array_2d = multiview_pts_2d
    # clean_point_indices = np.arange(multiview_pts_2d.shape[1])

    # Clean up nans
    nan_rows = np.isnan(multiview_pts_2d).any(axis=-0).any(axis=-1)
    pts_all_flat = np.arange(multiview_pts_2d.shape[1])
    pts_array_2d = multiview_pts_2d[:, ~nan_rows, :]
    clean_point_indices = pts_all_flat[~nan_rows]

    info_dict = {}
    info_dict["num_frames"] = num_frames
    info_dict["num_analyzed_body_parts"] = num_analyzed_body_parts
    info_dict["num_cameras"] = num_cameras
    info_dict["num_points_all"] = num_points_all
    info_dict["clean_point_indices"] = clean_point_indices

    # Get path images
    multivew_images_dir = Path("./data/Sawtell-data/images").resolve()
    view_dirs = os.listdir(multivew_images_dir)
    # NOTE: assumes folders for each view are ordered the same as in image_settings.json
    # i.e. view_0 corresponds to height_lims[0]

    path_images = []
    for i in range(num_cameras):
        for view in view_dirs:
            if str(i) in view:
                view_images = os.listdir(multivew_images_dir / view)
                view_images = sorted_alphanumeric(view_images)
                view_images = [
                    str(multivew_images_dir / Path(view) / i) for i in view_images
                ]
                path_images.append(view_images)

    # Get image dimensions
    from PIL import Image

    img_heights = []
    img_widths = []
    for i in range(num_cameras):
        img = Image.open(path_images[i][0])
        width, height = img.size
        img_heights.append(height)
        img_widths.append(width)

    # Get focal lengths
    focal_lengths = []
    focal_length_mm = 15
    sensor_size = 12
    for i in range(num_cameras):
        # focal_length = (focal_length_mm * img_widths[i]) / sensor_size
        # focal_length = (focal_length_mm * img_heights[i]) / sensor_size
        focal_length = (
            focal_length_mm * ((img_heights[i] ** 2 + img_widths[i] ** 2) ** (1 / 2))
        ) / sensor_size

        focal_lengths.append(focal_length)

    return {
        "img_width": img_widths,
        "img_height": img_heights,
        "pts_array_2d": pts_array_2d,
        "info_dict": info_dict,
        "path_images": path_images,
        "focal_length": focal_lengths,
    }


if __name__ == "__main__":
    get_data()