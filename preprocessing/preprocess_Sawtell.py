from __future__ import print_function
import os
import numpy as np
from pathlib import Path
import sys
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.resolve()))
import commentjson
import argparse
from tqdm import tqdm
from time import time

# ToDo: make more general!! especially paths.


def get_data(
    img_settings,
    dlc_file,
    save_arrays=False,
    chunksize=None,
    bp_to_keep=None,
    start_idx=0,
    nrows=None,
):

    # data_dir is e.g., Joao's folder with .json, folders per view, and a .csv dlc file
    # data_dir = Path(data_dir).resolve()  # assuming you run from preprocessing folder
    # filename = data_dir / "tank_dataset_5.h5"
    # f = h5py.File(filename, "r")
    # print("-------------DATASET INFO--------------")
    # print("keys: ", f.keys())
    # print("annotated: ", f["annotated"])
    # print("annotations: ", f["annotations"])
    # print("images: ", f["images"])
    # print("skeleton: ", f["skeleton"])
    # print("skeleton_names: ", f["skeleton_names"])
    # print("----------------------------------------")

    """
    img_settings = commentjson.load(
        open(str(img_settings_path), "r")
    )  # ToDo: the path in the json doesn't make sense
    """
    num_cameras = len(img_settings["height_lims"])

    # data_dir = Path(
    #     "/Users/Sunsmeister/Desktop/Research/Brain/MultiView/3D-Animal-Pose/data/Sawtell-data/fish_tracking"
    # ).resolve()
    print("Reading CSV...")
    if chunksize is None or nrows is not None:
        dlc_data = pd.read_csv(dlc_file, skiprows=range(1, start_idx), nrows=nrows)
    else:
        dlc_data = pd.read_csv(
            dlc_file, chunksize=chunksize, skiprows=range(1, start_idx)
        )
        dlc_data = pd.concat([i for i in tqdm(dlc_data)], ignore_index=True)

    worm_colnames = dlc_data.columns[dlc_data.columns.str.contains("worm")]
    dlc_data.columns = dlc_data.columns.str.replace("worm_right_", "worm_1_right_")
    dlc_data.columns = dlc_data.columns.str.replace("worm_front_", "worm_1_")
    dlc_data.columns = dlc_data.columns.str.replace("worm_right", "worm_")
    dlc_data.columns = dlc_data.columns.str.replace("worm_front", "worm_")
    worm_colnames = dlc_data.columns[dlc_data.columns.str.contains("worm")]

    # points are (num_frames, 3 * num_bodyparts)
    dlc_points = np.asarray(dlc_data)[:, 1:]
    # Find x,y columns
    columns = list(dlc_data.columns)[1:]
    x_points = []
    y_points = []
    confidences = []
    skeleton_names = []
    for i, name in enumerate(columns):
        # return a length-108 list whose entries are 1D np.arrays
        if name.endswith("x"):
            x_points.append(dlc_points[:, i])
            skeleton_names.append(name[:-2])
        elif name.endswith("y"):
            y_points.append(dlc_points[:, i])
        elif name.endswith("confidence"):
            confidences.append(dlc_points[:, i])
    x_points = np.asarray(x_points).transpose()[:, :, np.newaxis]
    y_points = np.asarray(y_points).transpose()[:, :, np.newaxis]
    confidences = np.asarray(confidences).transpose()

    pts_array = np.concatenate((x_points, y_points), axis=-1)

    # Make Nans if low confidence:
    pts_array[confidences < 0.5] = np.nan

    # Get number of bodyparts
    num_analyzed_body_parts = int(pts_array.shape[1] / num_cameras)

    skeleton_names = np.asarray(skeleton_names)
    reindexing = np.argsort(skeleton_names)
    skeleton_names = np.sort(skeleton_names)
    pts_array = pts_array[:, reindexing, :]

    # NOTE: the order should match the order in `image_settings.json`
    view_names = ["top", "main", "right"]
    assert len(view_names) == num_cameras

    # NOTE: Empty list keeps all bodyparts
    if bp_to_keep == None:
        bp_to_keep = []

    # Maps view_name to index that has a bp in that view
    multiview_name_to_idx = {}
    for view_name in view_names:
        multiview_name_to_idx[view_name] = []

    new_skeleton_names = []
    bodypart_names_without_view = []

    for idx, name in enumerate(
        skeleton_names
    ):  # was prev f["skeleton_names"] building on the labels data.
        name_without_view = "_".join(name.split("_")[:-1])
        if len(bp_to_keep) > 0:
            # NOTE: Handling worm specially since we want to turn the 5 keypoints into 1
            if name_without_view not in bp_to_keep and (
                "worm" not in name_without_view or "worm" not in bp_to_keep
            ):
                continue

        new_skeleton_names.append(name)

        if name_without_view not in bodypart_names_without_view:
            bodypart_names_without_view.append(name_without_view)

        for view_name in view_names:
            if view_name == name.split("_")[-1]:  # name.decode("UTF-8").split("_")[-1]:
                multiview_name_to_idx[view_name].append(idx)

    if len(bp_to_keep) > 0:
        num_analyzed_body_parts = int(len(new_skeleton_names) / num_cameras)

    # (views, frames, bp, 2)
    pts_array_2d_joints = np.empty(
        shape=(num_cameras, pts_array.shape[0], num_analyzed_body_parts, 2)
    )

    confidences_bp = np.empty(
        shape=(num_cameras, pts_array.shape[0], num_analyzed_body_parts)
    )

    # Here, the views are out of order
    for i, view_name in enumerate(view_names):
        # Select rows from indices
        view_indices = multiview_name_to_idx[view_name]
        view_points = pts_array[:, view_indices, :]
        view_points[:, :, 0] -= img_settings["width_lims"][i][0]
        view_points[:, :, 1] -= img_settings["height_lims"][i][0]
        pts_array_2d_joints[i, :, :, :] = view_points

        confidences_bp[i, :, :] = confidences[:, view_indices]

    # pts_array_2d_joints = pts_array_2d_joints[:, :, reindexing, :]

    # ----------------------------------------------
    # Replace worm with median
    if "worm" in bp_to_keep:
        print("REPLACING WORM WITH MEDIAN")
        worm_indices = []
        for i, bp in enumerate(bodypart_names_without_view):
            if "worm" in bp:
                worm_indices.append(i)

        worm_indices = np.asarray(worm_indices, dtype=np.int32)
        pts_array_2d_joints[:, :, worm_indices, :] = np.nanmedian(
            pts_array_2d_joints[:, :, worm_indices, :], axis=2
        )[:, :, np.newaxis, :]
        pts_array_2d_joints = np.delete(pts_array_2d_joints, worm_indices[1:], axis=2)

        confidences_bp[:, worm_indices, :] = np.nanmedian(
            confidences_bp[:, worm_indices, :], axis=1
        )[:, np.newaxis, :]
        confidences_bp = np.delete(confidences_bp, worm_indices[1:], axis=2)

    # ----------------------------------------------

    """
    points_path = data_dir / "2d_points_array.npy"
    conf_path = data_dir / "2d_confidences_array.npy"

    # Write points
    if save_arrays:
        np.save(points_path, pts_array_2d_joints)
        np.save(conf_path, confidences_bp)
    """

    return pts_array_2d_joints, confidences_bp, img_settings


def get_args():
    parser = argparse.ArgumentParser(description="Sawtell preprocessing")
    # dataset
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Users/Sunsmeister/Desktop/Research/Brain/MultiView/3D-Animal-Pose/data/Sawtell-data/20201102_Joao",
    )
    parser.add_argument(
        "--image_settings",
        type=str,
        default="/Users/Sunsmeister/Desktop/Research/Brain/MultiView/3D-Animal-Pose/data/Sawtell-data/20201102_Joao/image_settings.json",
    )
    parser.add_argument(
        "--dlc_file",
        type=str,
        default="/Volumes/sawtell-locker/C1/free/vids/20201102_Joao/concatenated_tracking.csv",
    )
    parser.add_argument("--save_arrays", action="store_true")
    parser.add_argument("--chunksize", type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    get_data(
        img_settings_path=args.image_settings,
        dlc_file=args.dlc_file,
        save_arrays=args.save_arrays,
        chunksize=args.chunksize,
    )
