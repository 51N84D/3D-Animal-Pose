from pathlib import Path
import argparse
from preprocessing.process_config import read_yaml
from utils.utils_BA import clean_nans
import numpy as np
from utils.points_to_dataframe import points3d_arr_to_df, reprojection_errors_arr_to_df
from utils.utils_IO import combine_images, write_video
from utils.utils_plotting import draw_circles, drawLine, skew, plot_cams_and_points
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from utils.arr_utils import slice_high_confidence
import multiprocessing as mp
import time
from functools import partial
import pickle
from copy import deepcopy


def get_args():
    parser = argparse.ArgumentParser(description="3d reconstruction")

    # dataset
    parser.add_argument("--config", type=str)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="which frame to start with",
    )
    parser.add_argument(
        "--num_ba_frames",
        default=5000,
        type=int,
        help="how many top confidence frames to use",
    )
    parser.add_argument(
        "--num_triang_frames",
        type=int,
        default=1000,
        help="Number of frames per partition for triangulation",
    )

    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Downsampling for plotting",
    )

    parser.add_argument(
        "--csv_type",
        type=str,
        default="dlc",
        help="Format of CSV in config",
    )

    parser.add_argument(
        "--save_bad_frames",
        action="store_true",
        help="Save frames with high reprojection errors",
    )

    parser.add_argument(
        "--reproj_thresh",
        type=float,
        default=2,
        help="Threshold for reprojection errors to select bad frames",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=186450,
        help="Starting frame index for triangulation",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=2000,
        help="Number of frames to triangulate",
    )
    # -------------------- Optional Dataset-dependent arguments -----------------------
    return parser.parse_args()

def find_str_index_in_list(test_list: list, test_str: str) -> int:
    return np.where(np.asarray(test_list) == test_str)[0][0]

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
    config=None,
    write_frames=False,
):
    """Write parameters to file"""
    reproj_frames = []
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
            config=config,
        )
        if og_dims is None:
            combined_proj = combine_images(reproj_images, equal_size=True)
        else:
            assert img_settings is not None
            combined_proj = combine_images_og(reproj_images, og_dims, img_settings)
        if write_frames:
            cv2.imwrite(str(plot_dir / f"reproj_{frame_i}.jpg"), combined_proj)
        reproj_frames.append(combined_proj)
    return reproj_frames


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


def filter_view_points(points_2d_og):
    num_views = points_2d_og.shape[0]
    return np.sum(np.any(np.isnan(points_2d_og), axis=-1), axis=0) > (num_views - 2)


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
    filter_points=False,
    config=None,
):
    if len(path_images) > 2:
        points_filter = filter_view_points(points_2d_og)

    else:
        points_filter = filter_zero_view_points(points_2d_og)

    images = []
    # For each camera
    for cam_num in range(points_2d_reproj.shape[0]):
        cam1_points_2d_og = points_2d_og[0, i, :, :]

        if is_arr:
            img_path = path_images[cam_num][i]
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

        # draw skeleton lines
        for ind, names in enumerate(config["skeleton"]):
            pt1 = points_2d_og[cam_num, i, find_str_index_in_list(config["bp_names"], names[0]), :]  # (x,y) coords
            pt2 = points_2d_og[cam_num, i, find_str_index_in_list(config["bp_names"], names[1]), :]  # same
            if not (np.isnan(pt1)).any() and not (np.isnan(pt2)).any():  # draw line only if there are no nans
                cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 255, 255), 1, cv2.LINE_AA)

        draw_circles(
            img,
            points_filled.astype(np.int32),
            color_list,
            point_size=point_sizes[cam_num] * 2,
            marker_type="cross",
        )

        draw_circles(
            img,
            curr_points_2d_og.astype(np.int32),
            color_list,
            point_size=point_sizes[cam_num],
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


def triangulate_serial(points_3d, points_2d_split, cam_group):
    subset_length = points_2d_split[0].shape[1]
    for i, points_2d_subset in enumerate(points_2d_split):
        subset_points_3d = cam_group.triangulate_progressive(points_2d_subset)
        points_3d_start = i * subset_length
        points_3d_end = points_3d_start + points_2d_subset.shape[1]
        points_3d[points_3d_start:points_3d_end, :, :] = subset_points_3d
    return points_3d


def triangulate_multiprocess(points_3d, points_2d_split, cam_group):
    # points_3d = np.empty((points_2d_joints.shape[1], points_2d_joints.shape[2], 3))
    manager = mp.Manager()
    points_3d_list = manager.list()
    subset_length = points_2d_split[0].shape[1]
    p = mp.Pool(processes=3)
    func = partial(triangulate_subsets, cam_group)
    iter_list = [
        (points_3d_list, i, points_2d_subset, subset_length)
        for i, points_2d_subset in enumerate(points_2d_split)
    ]
    points_3d_list = p.map(func, iter_list)
    points_3d = merge_triang_results(points_3d_list, points_3d, subset_length)

    return points_3d


def triangulate_subsets(cam_group, iter_list):
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
    for view_idx in range(views):
        frame_views = []
        for frame in frames:
            frame_views.append(
                frame[
                    height_lims[view_idx][0] : height_lims[view_idx][1],
                    width_lims[view_idx][0] : width_lims[view_idx][1],
                    :,
                ]
            )
        new_frames.append(frame_views)

    return new_frames


def get_points_at_frame(points_3d, points_2d_joints, i=0, filtered=True):

    """
    if filtered:
        BA_array_3d_back = refill_nan_array(points_3d, info_dict, dimension="3d")
        BA_dict = ordered_arr_3d_to_dict(BA_array_3d_back, info_dict)
        slice_3d = np.asarray(
            [BA_dict["x_coords"][i], BA_dict["y_coords"][i], BA_dict["z_coords"][i]]
        ).transpose()
    """

    # Points shape (num_frames, num_bp, 3)

    slice_3d = points_3d[i, :, :]
    if filtered:
        filter_indices = filter_zero_view_points(points_2d_joints)
        slice_3d[filter_indices[i]] = np.nan

    return slice_3d


def save_skeleton(points_3d, config, cam_group, points_2d_joints, plot_dir):
    """Write parameters to file"""

    xe = 0
    ye = -1
    ze = -2
    total_rot = 2 * np.pi
    ind_start = 0
    ind_end = points_3d.shape[0] - 1
    downsampling = 10
    color_list = config.color_list
    plot_dir = Path(plot_dir) / "3D_plots"

    # keep every n frames
    rotations = np.arange(
        start=0, stop=total_rot, step=total_rot / (ind_end - ind_start) * downsampling
    )

    for frame_i in tqdm(range(ind_start, ind_end)):
        if frame_i % downsampling != 0:
            continue

        plot_dir.mkdir(exist_ok=True, parents=True)
        slice_3d = get_points_at_frame(points_3d, points_2d_joints, i=frame_i)

        if np.all(np.isnan(slice_3d)):
            slice_3d = np.zeros((1, 3))
            skel_fig = plot_cams_and_points(
                cam_group=None,
                points_3d=slice_3d,
                point_size=0,
                skeleton_bp=None,
                skeleton_lines=None,
                point_colors=color_list,
                show_plot=False,
            )
        else:
            skeleton_bp, skeleton_lines = get_skeleton_parts(slice_3d, config)
            skel_fig = plot_cams_and_points(
                cam_group=None,
                points_3d=slice_3d,
                point_size=5,
                skeleton_bp=skeleton_bp,
                skeleton_lines=skeleton_lines,
                font_size=10,
                point_colors=color_list,
                show_plot=False,
            )

        scene_camera = dict(
            up=dict(x=0, y=-1, z=0),
            # center=dict(x=0, y=0.1, z=0),
            eye=dict(x=xe, y=ye, z=ze),
        )
        cam_rot = R.from_rotvec([0, rotations[1], 0]).as_matrix()
        new_cam = np.matmul(cam_rot, np.asarray([xe, ye, ze]))
        xe, ye, ze = new_cam

        skel_fig.update_layout(title={"text": f"3D Points at Frame {frame_i}"})

        padding = []
        scaling = 0.5
        padding.append(np.nanmax(points_3d[:, 0]) - np.nanmin(points_3d[:, 0]))
        padding.append(np.nanmax(points_3d[:, 1]) - np.nanmin(points_3d[:, 1]))
        padding.append(np.nanmax(points_3d[:, 2]) - np.nanmin(points_3d[:, 2]))
        padding = np.asarray(padding) * scaling

        x_min = np.nanmin(points_3d[:, 0])
        x_max = np.nanmax(points_3d[:, 0])

        y_min = np.nanmin(points_3d[:, 1])
        y_max = np.nanmax(points_3d[:, 1])

        z_min = np.nanmin(points_3d[:, 2])
        z_max = np.nanmax(points_3d[:, 2])

        x_min = -1
        x_max = 1
        y_min = -1
        y_max = 1
        z_min = -4
        z_max = 4

        skel_fig.update_layout(
            scene=dict(
                xaxis=dict(
                    range=[
                        x_min,
                        x_max,
                    ]
                ),
                yaxis=dict(
                    range=[
                        y_min,
                        y_max,
                    ]
                ),
                zaxis=dict(
                    range=[
                        z_min,
                        z_max,
                    ]
                ),
            )
        )

        skel_fig["layout"]["uirevision"] = "nothing"
        skel_fig["layout"]["scene"]["aspectmode"] = "cube"
        skel_fig.update_layout(scene_camera=scene_camera)

        skel_fig.write_image(str(plot_dir / f"3dpoints_{frame_i}.png"))


def get_skeleton_parts(slice_3d, config):

    skeleton_bp = {}

    for i in range(slice_3d.shape[0]):
        skeleton_bp[config.bp_names[i]] = tuple(slice_3d[i, :])
    skeleton_lines = config.skeleton

    return skeleton_bp, skeleton_lines


def reconstruct_points(
    config,
    output_dir,
    num_ba_frames=5000,
    num_triang_frames=1000,
    downsample=1,
    csv_type="dlc",
    start_idx=0,
    nrows=None,
    chunksize=10000,
    save_bad_frames=True,
    reproj_thresh=2,
    start_idx=0,
    nrows=None
):

    experiment_data = read_yaml(
        config, csv_type=csv_type, start_idx=start_idx, nrows=nrows
    )
    cam_group = experiment_data["cam_group"]
    num_frames = experiment_data["num_frames"]
    num_bodyparts = experiment_data["num_bodyparts"]
    config = experiment_data["config"]
    num_cameras = experiment_data["num_cams"]
    point_sizes = experiment_data["point_sizes"]
    if isinstance(point_sizes, int):
        point_sizes = [point_sizes] * num_cameras

    config = experiment_data["config"]
    color_list = config.color_list
    F = get_F_geometry(cam_group)
    confs = experiment_data["likelihoods"]
    points_2d_joints = experiment_data["points_2d_joints"]
    img_settings = experiment_data["img_settings"]
    video_paths = experiment_data["video_paths"]
    image_heights = experiment_data["image_heights"]
    image_widths = experiment_data["image_widths"]

    output_dir = Path(output_dir).resolve()

    if points_2d_joints.shape[0] > 2:
        points_filter = filter_view_points(points_2d_joints)

    else:
        points_filter = filter_zero_view_points(points_2d_joints)

    np.save("./ibl_points_2d.npy", points_2d_joints) # TODO: remove, specific
    # Select the top 50% frames with highest likelihood
    num_high_conf = points_2d_joints.shape[1] // 2
    if num_high_conf < num_ba_frames:
        num_ba_frames = num_high_conf
    pts_2d_high_conf = slice_high_confidence(points_2d_joints, confs, num_high_conf)

    # Randomly subsample
    pts_2d_high_conf_samples = pts_2d_high_conf[
        :,
        np.random.choice(pts_2d_high_conf.shape[1], num_ba_frames, replace=False),
        :,
        :,
    ]
    # num_ba_frames

    pts_2d_ba = pts_2d_high_conf_samples.reshape(
        pts_2d_high_conf_samples.shape[0],
        pts_2d_high_conf_samples.shape[1] * pts_2d_high_conf_samples.shape[2],
        pts_2d_high_conf_samples.shape[3],
    )

    # #      adjust points
    # points_2d_joints = experiment_data["points_2d_joints"]
    # pts_2d = points_2d_joints.reshape(
    #     points_2d_joints.shape[0],
    #     points_2d_joints.shape[1] * points_2d_joints.shape[2],
    #     points_2d_joints.shape[3],
    # )
    # Some Nan's may still remain if the fraction of high_conf/total frames is closer to 1.

    pts_2d_filtered, clean_point_indices = clean_nans(pts_2d_ba)
    print("####################################")
    print("entering vanilla bundle adjust...")
    print("####################################")

    res, points_3d_init = cam_group.bundle_adjust(pts_2d_filtered)

    # Triangulate points

    # ToDo: points_2d_joints should be splitted into chunks, with e.g,. 10 parallel calls to triangulate_progressive.
    print("####################################")
    print("entering triangulate_progressive...")
    print("####################################")

    if points_2d_joints.shape[1] < num_triang_frames:
        num_triang_frames = points_2d_joints.shape[1] + 1
        num_partition = 1
    else:
        num_partition = int(points_2d_joints.shape[1] / num_triang_frames)

    print("num_partition: ", num_partition)
    r = points_2d_joints.shape[1] - (num_triang_frames * num_partition)
    if r > 0:
        points_2d_split = np.split(
            points_2d_joints[:, : num_triang_frames * num_partition, :, :],
            num_partition,
            axis=1,
        )
        points_2d_split.append(
            points_2d_joints[:, num_triang_frames * num_partition :, :, :]
        )
    else:
        points_2d_split = np.split(points_2d_joints, num_partition, axis=1)

    points_3d = np.empty((points_2d_joints.shape[1], points_2d_joints.shape[2], 3))

    start_time = time.time()
    points_3d = triangulate_multiprocess(points_3d, points_2d_split, cam_group)
    end_time = time.time()

    print("triangulation time: ", end_time - start_time)
    print("####################################")
    print("triangulation finished!")
    print("####################################")

    if len(points_3d.shape) < 3:
        points_3d = points_3d.reshape(num_frames, num_bodyparts, 3)
    # Save points
    # ToDo: change paths to save in a designated folder
    output_dir.mkdir(parents=True, exist_ok=True)

    # ToDo: this should be reshaped into the original dlc format (e.g., just the relevant columns).
    df = points3d_arr_to_df(points_3d, config.bp_names)
    df.to_csv(output_dir / "points_3d.csv")

    print("Finished saving points.")

    # ToDo: Dan will add a final repreoject_points() call and save a csv that looks like the original 2D one.

    points_2d_reproj = reproject_points(points_3d, cam_group)
    print("points_2d_reproj: ", points_2d_reproj.shape)
    # TODO: consolidate all the start_idx and nrows in the script

    frame_indices = np.arange(start_idx, start_idx + nrows, step=downsample)
    array_indices = np.arange(points_2d_joints.shape[1], step=downsample) # previous implementation. this was focused on downsampling

    save_dict = {}
    save_dict["predictions"] = {}
    save_dict["reprojections"] = {}
    save_dict["BA"] = {}
    save_dict["points_3d"] = points_3d
    save_dict["cam_group"] = cam_group
    save_dict["config"] = config
    save_dict["start_idx"] = start_idx
    save_dict["nrows"] = nrows

    print("-------------------SAVING AS DICT---------------------")
    for view_idx in range(points_2d_joints.shape[0]):
        view_name = config.view_names[view_idx]
        save_dict["predictions"][view_name] = {}
        save_dict["reprojections"][view_name] = {}

        for bp_idx, bp in enumerate(config.bp_names):
            save_dict["predictions"][view_name][bp] = points_2d_joints[
                view_idx, :, bp_idx
            ]
            save_dict["reprojections"][view_name][bp] = points_2d_reproj[
                view_idx, :, bp_idx
            ]

            if view_idx == 0:
                save_dict["BA"][bp] = points_3d[:, bp_idx, :]

    # Store data (serialize)
    # TODO: results name use naive
    with open(output_dir / "reconstruction_results.pickle", "wb") as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("------------------------------------------------------")
    # NOTE: Not saving reprojections here

    print("####################################")
    print("extracting frames")
    print("####################################")

    if config.mirrored:
        frames = extract_frames(frame_indices, video_paths[0])
        og_dims = frames[0].shape
        frames = cut_frames(frames, img_settings)
    else:
        og_dims = None
        img_settings = None
        frames = []
        for vid_path in video_paths:
            frames.append(extract_frames(frame_indices, vid_path))

    
    reproj_frames = save_reproj(
        points_2d_reproj[:, array_indices, :, :],
        points_2d_joints[:, array_indices, :, :],
        cam_group,
        point_sizes,
        F,
        color_list,
        frames,
        output_dir / "reprojections",
        og_dims,
        img_settings,
        config,
    )
    write_video(
        frames=reproj_frames,
        out_file=str(output_dir / "reprojections.mov"),
        fps=10,
        add_text=True,
    )
    # -------------Get reprojection errors-----------------

    # Getting frames with high reprojection error
    '''
    for i in range(points_2d_reproj.shape[0]):
        points_2d_reproj[i] = np.mod(
            points_2d_reproj[i],
            np.asarray([image_widths[i], image_heights[i]])[
                np.newaxis, np.newaxis, np.newaxis, :
            ],
        )

    reprojection_errors = np.abs(points_2d_reproj - points_2d_joints)
    reproj_error_df = reprojection_errors_arr_to_df(
        reprojection_errors, config.bp_names
    )
    reproj_error_df.to_csv(output_dir / "reprojection_errors.csv")
    average_reproj_errors = np.nanmean(
        np.nanmean(np.nanmean(reprojection_errors, axis=0), axis=-1), axis=-1
    )

    # plt.hist(average_reproj_errors, bins=100)
    # plt.show()
    bad_reproj_indices = np.where(average_reproj_errors > reproj_thresh)[0]
    # frames_filter = np.all(points_filter, axis=-1)
    # bad_reproj_indices = np.where(~frames_filter[bad_reproj_indices])[0]
    # -------------------------------------------------------

    # save_skeleton(points_3d, config, cam_group, points_2d_joints, output_dir)
    '''

    """
    print("Writing bad frames...")
    if config.mirrored:
        frames = extract_frames(bad_reproj_indices, video_paths[0])
        og_dims = frames[0].shape
        frames = cut_frames(frames, img_settings)
    else:
        frames = []
        for vid_path in video_paths:
            frames.append(extract_frames(bad_reproj_indices, vid_path))

    reproj_frames_bad = save_reproj(
        points_2d_reproj[:, bad_reproj_indices, :, :],
        points_2d_joints[:, bad_reproj_indices, :, :],
        cam_group,
        point_sizes,
        F,
        color_list,
        frames,
        output_dir / "bad_reprojections",
        og_dims,
        img_settings,
        write_frames=True,
    )
    write_video(
        frames=reproj_frames_bad,
        out_file=str(output_dir / "bad_reprojections.mov"),
        fps=10,
        add_text=True,
    )

    print("Writing sorted frames...")
    frame_indices = np.arange(points_2d_joints.shape[1], step=1)

    if config.mirrored:
        frames = extract_frames(frame_indices, video_paths[0])
        og_dims = frames[0].shape
        frames = cut_frames(frames, img_settings)
    else:
        og_dims = None
        img_settings = None
        frames = []
        for vid_path in video_paths:
            frames.append(extract_frames(frame_indices, vid_path))

    reproj_frames = save_reproj(
        points_2d_reproj[:, frame_indices, :, :],
        points_2d_joints[:, frame_indices, :, :],
        cam_group,
        point_sizes,
        F,
        color_list,
        frames,
        output_dir / "reprojections",
        og_dims,
        img_settings,
    )

    reproj_sort_indices = np.argsort(average_reproj_errors)[::-1]
    reproj_frames_sorted = list(np.asarray(reproj_frames)[reproj_sort_indices])
    write_video(
        frames=reproj_frames_sorted,
        out_file=str(output_dir / "reprojections_sorted.mov"),
        fps=10,
        add_text=True,
    )
    """


if __name__ == "__main__":
    args = get_args()
    reconstruct_points(
        config=args.config,
        output_dir=args.output_dir,
        num_ba_frames=args.num_ba_frames,
        num_triang_frames=args.num_triang_frames,
        downsample=args.downsample,
        csv_type=args.csv_type,
        save_bad_frames=args.save_bad_frames,
        start_idx=args.start_idx,
        nrows=args.nrows
    )
