from __future__ import print_function
import os
import numpy as np
from pathlib import Path
import re


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def get_data(eid="cb2ad999-a6cb-42ff-bf71-1774c57e5308", trial_range=[5, 7]):
    res_folder = (
        "/Users/Sunsmeister/Desktop/Research/Brain/MultiView/3D-Animal-Pose/data/IBL_example/%s_trials_%s_%s"
        % (eid, trial_range[0], trial_range[1])
    )

    XYs_left = np.load(res_folder + "/XYs_left.npy", allow_pickle=True).flatten()[0]
    XYs_right = np.load(res_folder + "/XYs_right.npy", allow_pickle=True).flatten()[0]

    times_left = np.load(res_folder + "/times_left.npy")
    times_right = np.load(res_folder + "/times_right.npy")

    # get closest stamps or right cam (150 Hz) for each stamp of left (60 Hz)
    idx_aligned = []
    for t in times_left:
        idx_aligned.append(find_nearest(times_right, t))

    # paw_l in video left = paw_r in video right
    # Divide left coordinates by 2 to get them in half resolution like right cam;
    # reduce temporal resolution of right cam to that of left cam
    num_analyzed_body_parts = 3  # both paws and nose

    cam_right_paw1 = np.array(
        [XYs_right["paw_r"][0][idx_aligned], XYs_right["paw_r"][1][idx_aligned]]
    )
    cam_left_paw1 = np.array([XYs_left["paw_l"][0] / 2, XYs_left["paw_l"][1] / 2])

    cam_right_paw2 = np.array(
        [XYs_right["paw_l"][0][idx_aligned], XYs_right["paw_l"][1][idx_aligned]]
    )
    cam_left_paw2 = np.array([XYs_left["paw_r"][0] / 2, XYs_left["paw_r"][1] / 2])

    cam_right_nose = np.array(
        [XYs_right["nose_tip"][0][idx_aligned], XYs_right["nose_tip"][1][idx_aligned]]
    )
    cam_left_nose = np.array([XYs_left["nose_tip"][0] / 2, XYs_left["nose_tip"][1] / 2])

    # the format shall be such that points are concatenated, p1,p2,p3,p1,p2,p3, ...
    cam1 = np.zeros((len(idx_aligned) * num_analyzed_body_parts, 2))
    cam1[0::3] = cam_right_paw1.T
    cam1[1::3] = cam_right_paw2.T
    cam1[2::3] = cam_right_nose.T

    cam2 = np.zeros((len(idx_aligned) * num_analyzed_body_parts, 2))
    cam2[0::3] = cam_left_paw1.T
    cam2[1::3] = cam_left_paw2.T
    cam2[2::3] = cam_left_nose.T

    pts_array_2d_with_nans = np.array([cam1, cam2])


    num_cameras, num_points_all, _ = pts_array_2d_with_nans.shape

    pts_array_2d_joints = pts_array_2d_with_nans.reshape(
        num_cameras, len(times_left), num_analyzed_body_parts, 2
    )

    np.save('/Users/Sunsmeister/Desktop/Research/Brain/MultiView/3D-Animal-Pose/data/ibl.npy', pts_array_2d_joints)

    # remove nans (any of the x_r,y_r, x_l, y_l) and keep clean_point_indices
    non_nan_idc = ~np.isnan(pts_array_2d_with_nans).any(axis=2).any(axis=0)

    info_dict = {}
    info_dict["num_frames"] = len(times_left)
    info_dict["num_cameras"] = num_cameras
    info_dict["num_analyzed_body_parts"] = num_analyzed_body_parts
    info_dict["num_points_all"] = num_points_all
    info_dict["clean_point_indices"] = np.arange(num_points_all)[non_nan_idc]

    pts_array_2d = pts_array_2d_with_nans[:, info_dict["clean_point_indices"]]

    IMG_WIDTH_1 = IMG_WIDTH_2 = 640
    IMG_HEIGHT_1 = IMG_HEIGHT_2 = 512

    left_path = Path(
        f"/Users/Sunsmeister/Desktop/Research/Brain/MultiView/3D-Animal-Pose/data/IBL_example/{eid}_trials_{trial_range[0]}_{trial_range[1]}/images/imgs_left"
    )
    right_path = Path(
        f"/Users/Sunsmeister/Desktop/Research/Brain/MultiView/3D-Animal-Pose/data/IBL_example/{eid}_trials_{trial_range[0]}_{trial_range[1]}/images/imgs_right"
    )
    left_frames = sorted_nicely(os.listdir(left_path))
    left_frames.append(left_frames[-1])
    left_frames = [left_path / i for i in left_frames]
    right_frames = sorted_nicely(os.listdir(right_path))
    right_frames.append(right_frames[-1])
    right_frames = [right_path / i for i in right_frames]

    path_images = [left_frames, right_frames]

    focal_length_mm = 16
    sensor_size = 12.7

    focal_length = focal_length_mm * IMG_WIDTH_1 / sensor_size

    return {
        "img_width": [IMG_WIDTH_1, IMG_WIDTH_2],
        "img_height": [IMG_HEIGHT_1, IMG_HEIGHT_2],
        "pts_array_2d": pts_array_2d,
        "info_dict": info_dict,
        "path_images": path_images,
        "focal_length": [focal_length] * info_dict["num_cameras"],
    }


if __name__ == "__main__":
    #eid = "cb2ad999-a6cb-42ff-bf71-1774c57e5308"
    eid = 'e5fae088-ed96-4d9b-82f9-dfd13c259d52'
    #trial_range = [5, 7]
    trial_range = [10,13]
    get_data(eid, trial_range)