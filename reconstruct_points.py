from pathlib import Path
import argparse
from preprocessing.process_config import read_yaml
from utils.utils_BA import clean_nans
import numpy as np
from utils.points_to_dataframe import points3d_arr_to_df
from utils.utils_IO import (
    combine_images,
)
from utils.utils_plotting import draw_circles, drawLine, skew
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def get_args():
    parser = argparse.ArgumentParser(description="3d reconstruction")

    # dataset
    parser.add_argument("--config", type=str, default="./configs/sawtell.yaml")
    parser.add_argument("--skip_frame", default=None, type=int, help="frame to skip")
    parser.add_argument(
        "--point_sizes",
        default=7,
        type=int,
        nargs="+",
        help="sizes of points in each view",
    )
    return parser.parse_args()


def reproject_points(points_3d, cam_group):

    multivew_filled_points_2d = []

    for cam in cam_group.cameras:
        points_2d = np.squeeze(cam.project(points_3d))
        points_2d = points_2d.reshape(num_frames, num_bodyparts, 2)
        multivew_filled_points_2d.append(points_2d)

    multivew_filled_points_2d = np.asarray(multivew_filled_points_2d)
    return multivew_filled_points_2d


def save_reproj(points_3d, cam_group, pts_2d_joints, point_sizes, F, color_list):
    """Write parameters to file"""
    points_2d_reproj = reproject_points(points_3d, cam_group)
    points_2d_og = pts_2d_joints
    print("points_2d_og: ", points_2d_og.shape)
    for frame_i in tqdm(range(points_2d_og.shape[1])):
        plot_dir = Path("./reprojections")
        plot_dir.mkdir(exist_ok=True, parents=True)

        reproj_images = get_reproject_images(
            points_2d_reproj, points_2d_og, path_images, point_sizes, F, color_list, i=frame_i
        )
        combined_proj = combine_images(reproj_images, equal_size=True)
        cv2.imwrite(str(plot_dir / f"reproj_{frame_i}.jpg"), combined_proj)


def filter_zero_view_points(points_2d_og):
    return np.all(np.any(np.isnan(points_2d_og), axis=-1), axis=0)


def get_reproject_images(
    points_2d_reproj, points_2d_og, path_images, point_sizes, F, color_list, i=0
):
    plot_epipolar = True
    points_filter = filter_zero_view_points(points_2d_og)

    images = []
    # For each camera
    for cam_num in range(len(path_images)):
        cam1_points_2d_og = points_2d_og[0, i, :, :]

        img_path = path_images[cam_num][i]
        img = plt.imread(img_path)

        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if np.max(img) <= 1:
            img *= 255

        curr_points_2d_og = points_2d_og[cam_num, i, :, :]
        curr_points_2d_reproj = points_2d_reproj[cam_num, i, :, :]

        curr_points_2d_reproj[points_filter[i, :]] = np.nan

        # Get indices of interpolated points
        nonfilled_indices = np.any(np.isnan(curr_points_2d_og), axis=-1)
        points_nonfilled = np.empty_like(curr_points_2d_reproj)
        points_nonfilled[:] = np.nan
        points_filled = np.empty_like(curr_points_2d_reproj)
        points_filled[:] = np.nan
        num_nans = np.sum(nonfilled_indices)

        points_nonfilled[nonfilled_indices] = curr_points_2d_reproj[nonfilled_indices]
        points_filled[~nonfilled_indices] = curr_points_2d_reproj[~nonfilled_indices]

        draw_circles(
            img,
            curr_points_2d_og.astype(np.int32),
            color_list,
            point_size=point_sizes[cam_num],
        )

        draw_circles(
            img,
            points_filled.astype(np.int32),
            color_list,
            point_size=point_sizes[cam_num] * 4,
            marker_type="cross",
        )

        draw_circles(
            img,
            points_nonfilled.astype(np.int32),
            color_list,
            point_size=point_sizes[cam_num],
            thickness=2,
        )

        # Draw epipolar lines
        if cam_num != 0 and plot_epipolar and num_nans > 0:
            cam1_points_2d_og = cam1_points_2d_og.astype(np.int32)
            cam1_points_2d_og = cam1_points_2d_og[
                ~np.isnan(cam1_points_2d_og).any(axis=1)
            ]
            ones = np.ones(cam1_points_2d_og.shape[0])[:, np.newaxis]
            points_homog = np.concatenate((cam1_points_2d_og, ones), axis=-1)

            aug_F = np.tile(F[cam_num - 1], (points_homog.shape[0], 1, 1))
            lines = np.squeeze(np.matmul(aug_F, points_homog[:, :, np.newaxis]))
            for j, line_vec in enumerate(lines):
                if not nonfilled_indices[j]:
                    continue
                # Get x and y intercepts (on image) to plot
                # y = 0: x = -c/a
                x_intercept = int(-line_vec[2] / line_vec[0])
                # x = 0: y = -c/b
                y_intercept = int(-line_vec[2] / line_vec[1])
                img = drawLine(img, x_intercept, 0, 0, y_intercept, color=color_list[j])

        images.append(img)

    return images


def get_F_geometry(cam_group):
    # Get rotation vector
    camera_1 = cam_group.cameras[0]
    rot_vec_1 = R.from_rotvec(camera_1.get_rotation())
    R1 = rot_vec_1.as_matrix()
    t1 = camera_1.get_translation()
    K1 = camera_1.get_camera_matrix()

    F_all = []

    for camera_2 in cam_group.cameras[1:]:
        # Get rotation vector
        rot_vec_2 = R.from_rotvec(camera_2.get_rotation())
        R2 = rot_vec_2.as_matrix()
        # Get translation vector
        t2 = camera_2.get_translation()
        K2 = camera_2.get_camera_matrix()

        # --- Now compute relevant quantities for F estimation ------
        # Camera matrix basics: http://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf
        # Fundamental matrix computation: https://rb.gy/dd0nz2

        # Compute projection matrices
        P1 = np.matmul(K1, np.concatenate((R1, t1[:, np.newaxis]), axis=1))
        P2 = np.matmul(K2, np.concatenate((R2, t2[:, np.newaxis]), axis=1))

        # Get camera center (view 1)
        R1_inv = np.linalg.inv(R1)
        C = np.matmul(-R1_inv, t1)
        C = np.append(C, 1)

        F = np.matmul(skew(np.matmul(P2, C)), np.matmul(P2, np.linalg.pinv(P1)))
        F_all.append(F)
    return F_all


if __name__ == "__main__":
    args = get_args()
    experiment_data = read_yaml(args.config, args.skip_frame)
    cam_group = experiment_data["cam_group"]
    num_frames = experiment_data["num_frames"]
    num_bodyparts = experiment_data["num_bodyparts"]
    config = experiment_data["config"]
    focal_length = experiment_data["focal_length"]
    path_images = experiment_data["frame_paths"]
    num_cameras = experiment_data["num_cams"]
    point_sizes = args.point_sizes
    if isinstance(point_sizes, int):
        point_sizes = [point_sizes] * num_cameras
    config = experiment_data["config"]
    color_list = config.color_list
    F = get_F_geometry(cam_group)

    # Bundle adjust points
    pts_2d_joints = experiment_data["points_2d_joints"]
    pts_2d = pts_2d_joints.reshape(
        pts_2d_joints.shape[0],
        pts_2d_joints.shape[1] * pts_2d_joints.shape[2],
        pts_2d_joints.shape[3],
    )
    pts_2d_filtered, clean_point_indices = clean_nans(pts_2d)
    res, points_3d_init = cam_group.bundle_adjust(pts_2d_filtered)

    # Triangulate points
    points_3d = cam_group.triangulate_progressive(pts_2d_joints)
    if len(points_3d.shape) < 3:
        points_3d = points_3d.reshape(num_frames, num_bodyparts, 3)
    # Save points
    points_dir = Path("./points").resolve()
    points_dir.mkdir(parents=True, exist_ok=True)

    df = points3d_arr_to_df(points_3d, config.bp_names)
    df.to_csv(points_dir / f"{Path(args.config).stem}_3d.csv")
    print("Finished saving points.")

    save_reproj(points_3d, cam_group, pts_2d_joints, point_sizes, F, color_list)
