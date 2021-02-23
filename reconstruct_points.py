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
from utils.arr_utils import slice_high_confidence
import multiprocessing as mp
import time


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
    parser.add_argument(
        "--num_ba_frames",
        default=1000,
        type=int,
        help="how many top confidence frames to use",
    )
    parser.add_argument("--save_reprojections", action="store_true")
    parser.add_argument(
        "--dataset", type=str, help="dataset name for unique preprocessing"
    )
    parser.add_argument("--chunksize", type=int, default=10000)
    parser.add_argument(
        "--num_triang_frames",
        type=int,
        default=2000,
        help="Number of frames per partition for triangulation",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="/Volumes/sawtell-locker/C1/free/vids/20201102_Joao/concatenated.avi",
    )

    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_save_frames", type=int, default=0)
    parser.add_argument("--downsample", type=int, default=1)

    return parser.parse_args()


def reproject_points(points_3d, cam_group):

    multivew_filled_points_2d = []
    (num_frames, num_bodyparts, _) = points_3d.shape

    for cam in cam_group.cameras:
        points_2d = np.squeeze(cam.project(points_3d))
        points_2d = points_2d.reshape(num_frames, num_bodyparts, 2)
        multivew_filled_points_2d.append(points_2d)

    multivew_filled_points_2d = np.asarray(multivew_filled_points_2d)
    return multivew_filled_points_2d


def save_reproj(
    points_2d_reproj,
    points_2d_og,
    cam_group,
    point_sizes,
    F,
    color_list,
    path_images,
    plot_dir,
    og_dims=None,
    img_settings=None,
):
    """Write parameters to file"""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    for frame_i in tqdm(range(points_2d_og.shape[1])):
        reproj_images = get_reproject_images(
            points_2d_reproj,
            points_2d_og,
            path_images,
            point_sizes,
            F,
            color_list,
            i=frame_i,
        )
        if og_dims is None:
            combined_proj = combine_images(reproj_images, equal_size=True)
        else:
            assert img_settings is not None
            combined_proj = combine_images_og(reproj_images, og_dims, img_settings)
        cv2.imwrite(str(plot_dir / f"reproj_{frame_i}.jpg"), combined_proj)


def combine_images_og(images, og_dims, config):
    image_canvas = np.zeros((og_dims[0], og_dims[1], 3), np.uint8)
    height_lims = config["height_lims"]
    width_lims = config["width_lims"]
    for i, img in enumerate(images):
        image_canvas[
            height_lims[i][0] : height_lims[i][1],
            width_lims[i][0] : width_lims[i][1],
            :,
        ] = img

    return image_canvas


def filter_zero_view_points(points_2d_og):
    return np.all(np.any(np.isnan(points_2d_og), axis=-1), axis=0)


def get_reproject_images(
    points_2d_reproj,
    points_2d_og,
    path_images,
    point_sizes,
    F,
    color_list,
    i=0,
    is_arr=True,
    plot_epipolar=False,
):
    points_filter = filter_zero_view_points(points_2d_og)

    images = []
    # For each camera
    for cam_num in range(points_2d_reproj.shape[0]):
        cam1_points_2d_og = points_2d_og[0, i, :, :]

        if is_arr:
            img_path = path_images[i][cam_num]
            img = img_path
        else:
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


def triangulate_serial(points_3d, points_2d_split):
    subset_length = points_2d_split[0].shape[1]
    for i, points_2d_subset in enumerate(points_2d_split):
        subset_points_3d = cam_group.triangulate_progressive(points_2d_subset)
        points_3d_start = i * subset_length
        points_3d_end = points_3d_start + points_2d_subset.shape[1]
        points_3d[points_3d_start:points_3d_end, :, :] = subset_points_3d
    return points_3d


def triangulate_multiprocess(points_3d, points_2d_split):
    # points_3d = np.empty((pts_2d_joints.shape[1], pts_2d_joints.shape[2], 3))
    manager = mp.Manager()
    points_3d_list = manager.list()
    subset_length = points_2d_split[0].shape[1]
    p = mp.Pool(processes=mp.cpu_count() - 1)
    iter_list = [
        (points_3d_list, i, points_2d_subset, subset_length)
        for i, points_2d_subset in enumerate(points_2d_split)
    ]

    points_3d_list = p.map(triangulate_subsets, iter_list)
    points_3d = merge_triang_results(points_3d_list, points_3d, subset_length)

    return points_3d


def triangulate_subsets(iter_list):
    points_3d_list, i, points_2d_subset, subset_length = iter_list
    subset_points_3d = cam_group.triangulate_optim(points_2d_subset)
    return (subset_points_3d, i)


def merge_triang_results(
    points_3d_list,
    points_3d,
    subset_length,
):
    for idx, (subset_points_3d, i) in enumerate(points_3d_list):
        points_3d_start = i * subset_length
        points_3d_end = points_3d_start + subset_points_3d.shape[0]
        points_3d[points_3d_start:points_3d_end, :, :] = subset_points_3d
    return points_3d


def extract_frames(indices, video_path, exact=True):
    cap = cv2.VideoCapture(video_path)
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frames = []
    if exact:
        indices_dict = dict((idx, 0) for idx in indices)
        curr_idx = 0
        while len(indices_dict.keys()) > 0 and curr_idx < totalFrames:
            ret, frame = cap.read()
            if curr_idx in indices_dict:
                frames.append(frame)
                del indices_dict[curr_idx]
            curr_idx += 1

    else:
        for frame_idx in indices:
            # get total number of frames
            # check for valid frame number
            if frame_idx >= 0 & frame_idx <= totalFrames:
                # set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                frames.append(frame)
    return frames


def cut_frames(frames, config):
    height_lims = config["height_lims"]
    width_lims = config["width_lims"]
    views = len(width_lims)
    new_frames = []
    for frame in frames:
        frame_views = []
        for view_idx in range(views):
            frame_views.append(
                frame[
                    height_lims[view_idx][0] : height_lims[view_idx][1],
                    width_lims[view_idx][0] : width_lims[view_idx][1],
                    :,
                ]
            )
        new_frames.append(frame_views)

    return new_frames


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

    # load pts
    if args.dataset == "Sawtell_Fish":
        from preprocessing.preprocess_Sawtell_DLC import get_data

        pts_2d_joints, confs, img_settings = get_data(
            data_dir="./data/Sawtell-data/20201102_Joao",
            img_settings_path="./data/Sawtell-data/20201102_Joao/image_settings.json",
            dlc_file="/Volumes/sawtell-locker/C1/free/vids/20201102_Joao/concatenated_tracking.csv",
            save_arrays=False,
            chunksize=args.chunksize,
        )
    else:
        pts_2d_joints = experiment_data['points_2d_joints']
    assert (args.start_idx + args.num_save_frames) < pts_2d_joints.shape[1]

    num_high_conf = pts_2d_joints.shape[1] // 2
    if num_high_conf < args.num_ba_frames:
        num_high_conf = args.num_ba_frames
    pts_2d_high_conf = slice_high_confidence(pts_2d_joints, confs, num_high_conf)

    # Randomly subsample
    pts_2d_high_conf_samples = pts_2d_high_conf[
        :,
        np.random.choice(pts_2d_high_conf.shape[1], args.num_ba_frames, replace=False),
        :,
        :,
    ]
    # args.num_ba_frames

    pts_2d = pts_2d_high_conf_samples.reshape(
        pts_2d_high_conf_samples.shape[0],
        pts_2d_high_conf_samples.shape[1] * pts_2d_high_conf_samples.shape[2],
        pts_2d_high_conf_samples.shape[3],
    )

    # #      adjust points
    # pts_2d_joints = experiment_data["points_2d_joints"]
    # pts_2d = pts_2d_joints.reshape(
    #     pts_2d_joints.shape[0],
    #     pts_2d_joints.shape[1] * pts_2d_joints.shape[2],
    #     pts_2d_joints.shape[3],
    # )
    # Some Nan's may still remain if the fraction of high_conf/total frames is closer to 1.
    pts_2d_filtered, clean_point_indices = clean_nans(pts_2d)
    print("####################################")
    print("entering vanilla bundle adjust...")
    print("####################################")

    res, points_3d_init = cam_group.bundle_adjust(pts_2d_filtered)

    # Triangulate points

    # ToDo: pts_2d_joints should be splitted into chunks, with e.g,. 10 parallel calls to triangulate_progressive.
    print("####################################")
    print("entering triangulate_progressive...")
    print("####################################")

    num_partition = int(pts_2d_joints.shape[1] / args.num_triang_frames)
    r = pts_2d_joints.shape[1] - (args.num_triang_frames * num_partition)
    if r > 0:
        points_2d_split = np.split(
            pts_2d_joints[:, : args.num_triang_frames * num_partition, :, :],
            num_partition,
            axis=1,
        )
        points_2d_split.append(
            pts_2d_joints[:, args.num_triang_frames * num_partition :, :, :]
        )
    else:
        points_2d_split = np.split(pts_2d_joints, num_partition, axis=1)

    points_3d = np.empty((pts_2d_joints.shape[1], pts_2d_joints.shape[2], 3))

    start_time = time.time()
    points_3d_mp = triangulate_multiprocess(points_3d, points_2d_split)
    end_time = time.time()

    print("triangulation time: ", end_time - start_time)

    print("####################################")
    print("triangulation finished!")
    print("####################################")

    if len(points_3d.shape) < 3:
        points_3d = points_3d.reshape(num_frames, num_bodyparts, 3)
    # Save points
    # ToDo: change paths to save in a designated folder
    points_dir = Path("./points").resolve()
    points_dir.mkdir(parents=True, exist_ok=True)

    # create reprojections
    points_reproj = reproject_points(points_3d, cam_group)

    # ToDo: this should be reshaped into the original dlc format (e.g., just the relevant columns).

    df = points3d_arr_to_df(points_3d, config.bp_names)
    df.to_csv(points_dir / f"{Path(args.config).stem}_3d.csv")
    print("Finished saving points.")

    # ToDo: Dan will add a final repreoject_points() call and save a csv that looks like the original 2D one.

    points_2d_reproj = reproject_points(points_3d, cam_group)

    indices = np.arange(
        start=args.start_idx,
        stop=args.start_idx + args.num_save_frames,
        step=args.downsample,
    )

    print("####################################")
    print("extracting frames")
    print("####################################")

    frames = extract_frames(indices, args.video_path)
    og_dims = frames[0].shape
    frames = cut_frames(frames, img_settings)

    if args.save_reprojections:
        save_reproj(
            points_2d_reproj[:, indices, :, :],
            pts_2d_joints[:, indices, :, :],
            cam_group,
            point_sizes,
            F,
            color_list,
            frames,
            "./reprojections",
            og_dims,
            img_settings,
        )

    # Write frames
