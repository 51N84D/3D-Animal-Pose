import h5py
import re
import os
from pathlib import Path
import numpy as np


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def get_data():
    data_dir = Path("./data/CostaData").resolve()

    # Read cam 1 points
    filename = (
        data_dir / "camera-1_09-15DeepCut_resnet50_Joystick_cam1Sep13shuffle1_140000.h5"
    )
    f = h5py.File(filename, "r")
    data_1 = f["df_with_missing"]["table"]
    # Hand is removed at image 1178, so index is 1 less:
    im_start = 1178
    arr_start_idx = im_start - 1
    data_1 = data_1[arr_start_idx:]
    data_1 = data_1["values_block_0"]

    # Read cam 2 points
    filename = (
        data_dir / "camera-2_09-15DeepCut_resnet50_Joystick_cam2Sep13shuffle1_160000.h5"
    )
    f = h5py.File(filename, "r")
    data_2 = f["df_with_missing"]["table"]
    # Hand is removed at image 1178, so index is 1 less:
    im_start = 1178
    arr_start_idx = im_start - 1
    data_2 = data_2[arr_start_idx:]
    data_2 = data_2["values_block_0"]

    # Set up path_images
    cam_1_image_path = data_dir / "cam_1_09_15_all_images/"
    cam_2_image_path = data_dir / "cam_2_09_15_all_images/"

    cam_1_images = sorted_nicely(os.listdir(cam_1_image_path))
    cam_2_images = sorted_nicely(os.listdir(cam_2_image_path))

    if ".DS_Store" in cam_1_images:
        cam_1_images.remove(".DS_Store")
    if ".DS_Store" in cam_2_images:
        cam_2_images.remove(".DS_Store")

    # Find index
    for idx, i in enumerate(cam_1_images):
        if str(im_start) in i:
            cam1_start_idx = idx
            break
    cam_1_images = cam_1_images[cam1_start_idx:]

    for idx, i in enumerate(cam_2_images):
        if str(im_start) in i:
            cam2_start_idx = idx
            break
    cam_2_images = cam_2_images[cam2_start_idx:]

    # Append paths to each filename
    cam_1_images = [cam_1_image_path / i for i in cam_1_images]
    cam_2_images = [cam_2_image_path / i for i in cam_2_images]

    assert len(cam_1_images) == len(cam_2_images)
    path_images = [cam_1_images, cam_2_images]

    # Ignore likelihoods, only take points. Columns to delete 2,5,8
    data_1 = np.delete(data_1, [2, 5, 8], 1)
    data_2 = np.delete(data_2, [2, 5, 8], 1)

    # Set up pts_array_2d
    # re-organize camera 1
    pts_2d_cam1_x = data_1[:, [0, 2, 4]]
    pts_2d_cam1_y = data_1[:, [1, 3, 5]]

    pts_2d_cam1_x = np.ravel(pts_2d_cam1_x)[:, np.newaxis]
    pts_2d_cam1_y = np.ravel(pts_2d_cam1_y)[:, np.newaxis]

    pts_2d_cam1 = np.concatenate((pts_2d_cam1_x, pts_2d_cam1_y), axis=-1)[
        np.newaxis, :, :
    ]

    # re-organize camera 2
    pts_2d_cam2_x = data_2[:, [0, 2, 4]]
    pts_2d_cam2_y = data_2[:, [1, 3, 5]]

    pts_2d_cam2_x = np.ravel(pts_2d_cam2_x)[:, np.newaxis]
    pts_2d_cam2_y = np.ravel(pts_2d_cam2_y)[:, np.newaxis]

    pts_2d_cam2 = np.concatenate((pts_2d_cam2_x, pts_2d_cam2_y), axis=-1)[
        np.newaxis, :, :
    ]

    # Now combine
    pts_array_2d = np.concatenate((pts_2d_cam1, pts_2d_cam2), axis=0)
    print(pts_array_2d.shape)

    # Set up info_dict:
    info_dict = {}

    num_frames = data_1.shape[0]
    info_dict["num_frames"] = num_frames

    num_analyzed_body_parts = 3
    info_dict["num_analyzed_body_parts"] = num_analyzed_body_parts

    num_cameras = 2
    info_dict["num_cameras"] = num_cameras

    num_points_all = pts_array_2d.shape[1]
    info_dict["num_points_all"] = num_points_all

    clean_point_indices = np.arange(pts_array_2d.shape[1])
    info_dict["clean_point_indices"] = clean_point_indices

    # Image size is 640x512
    P_X_1 = 320
    P_X_2 = 320

    P_Y_1 = 256
    P_Y_2 = 256

    return 0