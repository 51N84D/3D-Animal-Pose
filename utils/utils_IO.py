#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:02:42 2020

@author: danbiderman
mostly functions that transform the data between DLC form,
3D plotting form, and BA form.
"""
import pickle
import numpy as np
import matplotlib.image as mpimg
import cv2
import os
import re


# pickle utils
def save_object(obj, filename):
    with open(filename, "wb") as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, "rb") as input:  # note rb and not wb
        return pickle.load(input)


def pts_array_from_dlc_3d(dlc_mat):
    """'input: np.array [n_frames, n_body_parts X n_coords]
    output: np.array [n_frames X n_body_parts, n_coords]"""
    n_frames = dlc_mat.shape[0]
    n_pts_per_frame = int(
        dlc_mat.shape[1] / 3
    )  # for 3d, each point gets 3 cols (x,y,z)
    pts_array = np.zeros((n_frames * n_pts_per_frame, 3))  # 2 cols for x,y coords.
    counter = 0
    for i in np.arange(n_pts_per_frame) * 3:
        # print(i)
        pts_array[
            counter * dlc_mat.shape[0] : (counter + 1) * dlc_mat.shape[0], :
        ] = dlc_mat[:, [i, i + 1, i + 2]]
        counter += 1
    return pts_array


def pts_array_from_dlc(dlc_mat):
    """same as above but for 2d arrays."""
    n_frames = dlc_mat.shape[0]
    n_pts_per_frame = int(
        dlc_mat.shape[1] / 3
    )  # for 2d, each point gets 3 cols (x,y,log_prob)
    pts_array = np.zeros((n_frames * n_pts_per_frame, 2))  # 2 cols for x,y coords.
    counter = 0
    for i in np.arange(n_pts_per_frame) * 3:
        # print(i)
        pts_array[
            counter * dlc_mat.shape[0] : (counter + 1) * dlc_mat.shape[0], :
        ] = dlc_mat[:, [i, i + 1]]
        counter += 1
    return pts_array


def chop_dict(dict_to_chop, is3d, start_frame, n_frames):

    if is3d:
        key_names = ["x_coords", "y_coords", "z_coords"]
    else:
        key_names = ["x_coords", "y_coords"]

    temp_dict = {}
    for i in range(len(key_names)):
        temp_dict[key_names[i]] = dict_to_chop[key_names[i]][
            :, start_frame : (start_frame + n_frames)
        ]

    return temp_dict


def dict_to_arr(dict_3d):
    "convert dicts (in the form above) to arrays. "
    array_temp = np.hstack(
        [dict_3d["x_coords"].T, dict_3d["y_coords"].T, dict_3d["z_coords"].T]
    )
    return array_temp[:, [0, 3, 6, 1, 4, 7, 2, 5, 8]]


def arr3d_to_dict(arr):
    "convert n_frames by 9 array to a dict with x,y,z coords"
    dict_3d = {}
    dict_3d["x_coords"] = arr[:, [0, 3, 6]].T
    dict_3d["y_coords"] = arr[:, [1, 4, 7]].T
    dict_3d["z_coords"] = arr[:, [2, 5, 8]].T
    return dict_3d


def revert_ordered_arr_2d_to_dict(body_parts, n_cams, ordered_arr):
    """remove from here. in IO"""
    n_frames = int(ordered_arr.shape[0] / (body_parts * n_cams))
    coord_list_of_dicts = []

    for cam in range(n_cams):
        coord_dict = {}
        coord_dict["x_coords"] = np.zeros((body_parts, n_frames))
        coord_dict["y_coords"] = np.zeros((body_parts, n_frames))

        curr_cam_arr = ordered_arr[
            cam * n_frames * body_parts : (cam + 1) * n_frames * body_parts
        ]
        #         print(cam*n_frames*body_parts)
        #         print( (cam+1)*n_frames*body_parts)

        for f_ind in range(body_parts):
            coord_dict["x_coords"][f_ind, :] = curr_cam_arr[
                f_ind * n_frames : (f_ind + 1) * n_frames, 0
            ]
            coord_dict["y_coords"][f_ind, :] = curr_cam_arr[
                f_ind * n_frames : (f_ind + 1) * n_frames, 1
            ]
        coord_list_of_dicts.append(coord_dict)
    return coord_list_of_dicts


def revert_ordered_arr_to_dict(body_parts, ordered_arr):
    n_frames = int(ordered_arr.shape[0] / body_parts)
    coord_dict = {}
    coord_dict["x_coords"] = np.zeros((body_parts, n_frames))
    coord_dict["y_coords"] = np.zeros((body_parts, n_frames))
    coord_dict["z_coords"] = np.zeros((body_parts, n_frames))
    for f_ind in range(body_parts):
        coord_dict["x_coords"][f_ind, :] = ordered_arr[
            f_ind * n_frames : (f_ind + 1) * n_frames, 0
        ]
        coord_dict["y_coords"][f_ind, :] = ordered_arr[
            f_ind * n_frames : (f_ind + 1) * n_frames, 1
        ]
        coord_dict["z_coords"][f_ind, :] = ordered_arr[
            f_ind * n_frames : (f_ind + 1) * n_frames, 2
        ]
    return coord_dict


def ordered_arr_3d_to_dict(pts_array_3d, info_dict):
    """assuming you used np.flatten(). to create the 3d points.
    and assuming that the pts array is full size, including nans."""
    pose_dict = {}
    pose_dict["x_coords"] = pts_array_3d[:, 0].reshape(
        info_dict["num_frames"], info_dict["num_analyzed_body_parts"]
    )
    pose_dict["y_coords"] = pts_array_3d[:, 1].reshape(
        info_dict["num_frames"], info_dict["num_analyzed_body_parts"]
    )
    pose_dict["z_coords"] = pts_array_3d[:, 2].reshape(
        info_dict["num_frames"], info_dict["num_analyzed_body_parts"]
    )
    return pose_dict


def refill_nan_array(pts_array_clean, info_dict, dimension):
    """we take our chopped array and embedd it in a full array with nans"""

    num_cameras = info_dict["num_cameras"]

    if dimension == "3d":
        pts_refill = np.empty(
            (info_dict["num_frames"] * info_dict["num_analyzed_body_parts"], 3)
        )
        pts_refill[:] = np.NaN
        pts_refill[info_dict["clean_point_indices"], :] = pts_array_clean

    else:

        pts_all_flat = np.arange(
            info_dict["num_frames"] * info_dict["num_analyzed_body_parts"]
        )
        indices_init = np.concatenate([pts_all_flat] * num_cameras)

        pts_refill = np.empty(
            (
                info_dict["num_frames"]
                * info_dict["num_analyzed_body_parts"]
                * info_dict["num_cameras"],
                2,
            )
        )

        pts_refill[:] = np.NaN
        pts_refill[
            np.isin(indices_init, info_dict["clean_point_indices"]), :
        ] = pts_array_clean

    return pts_refill


def arr_2d_to_list_of_dicts(pts_array_2d, info_dict):
    """this works assuming that you flattened the x,y coords from the dataframe."""
    coord_list_of_dicts = []
    for cam in range(info_dict["num_cameras"]):

        row_indices = np.arange(
            cam * (pts_array_2d.shape[0] / info_dict["num_cameras"]),
            (cam + 1) * (pts_array_2d.shape[0] / info_dict["num_cameras"]),
        ).astype(int)

        coord_dict = {}
        coord_dict["x_coords"] = pts_array_2d[row_indices, 0].reshape(
            info_dict["num_frames"], info_dict["num_analyzed_body_parts"]
        )
        coord_dict["y_coords"] = pts_array_2d[row_indices, 1].reshape(
            info_dict["num_frames"], info_dict["num_analyzed_body_parts"]
        )

        coord_list_of_dicts.append(coord_dict)
    return coord_list_of_dicts


def read_image(image_path, flip):
    """if necessary, we can flip to make it easier to plot on top of the image."""
    im = mpimg.imread(image_path)
    if flip:
        im = np.flipud(im)
    return im


def make_image_array(img_indexes, flip):
    im_shape = read_image(img_indexes[0], flip).shape  # read first img
    num_frames = len(img_indexes)
    img_array = np.zeros((im_shape[0], im_shape[1], num_frames))
    print(img_array.shape)
    for i in range(num_frames):
        img_array[:, :, i] = read_image(img_indexes[i], flip)  # function defined above

    return img_array


# sorts incoming files alphanumerically into list
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


def read_frame_at_idx(video_path, frame_idx):
    vidcap = cv2.VideoCapture(str(video_path))
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if count >= frame_idx:
            return image
        count += 1

    if count < frame_idx:
        print(f"Request frame {frame_idx} is too large")
        return None


def write_video(
    frames=None, image_dir=None, out_file="./out.mp4", fps=5, add_text=False
):
    img_array = []

    if frames is not None:
        im_list = frames
        for img in im_list:
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

    if image_dir is not None:
        im_list = os.listdir(image_dir)
        im_list = sorted_alphanumeric(im_list)

        if ".DS_Store" in im_list:
            im_list.remove(".DS_Store")

        for filename in im_list:
            img = cv2.imread(os.path.join(image_dir, filename))
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

    out = cv2.VideoWriter(
        filename=out_file,
        fourcc=cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        fps=fps,
        frameSize=size,
    )  # 15 fps

    cv2.VideoWriter()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(len(img_array)):
        if add_text:
            cv2.putText(
                img_array[i], f"{i}", (10, 30), font, 1, (0, 0, 255), 1, cv2.LINE_AA
            )
        out.write(img_array[i])
    out.release()


def reproject_3d_points(points_3d, info_dict, pts_array_2d, cam_group):
    # now we first refill the full sized containers, then revert to dicts.

    # do the pts_array_3d_clean
    # pts_2d_orig
    pts_array_2d_og = np.reshape(
        pts_array_2d, (pts_array_2d.shape[0] * pts_array_2d.shape[1], -1)
    )

    num_cameras = len(cam_group.cameras)

    array_2d_orig = refill_nan_array(pts_array_2d_og, info_dict, dimension="2d")
    pose_list_2d_orig = arr_2d_to_list_of_dicts(array_2d_orig, info_dict)

    points_proj = []
    for cam in cam_group.cameras:
        points_proj.append(cam.project(points_3d).squeeze())

    points_proj = np.concatenate(points_proj, axis=0)

    # pts_2d_reproj
    array_2d_reproj_back = refill_nan_array(points_proj, info_dict, dimension="2d")
    pose_list_2d_reproj = arr_2d_to_list_of_dicts(array_2d_reproj_back, info_dict)

    joined_list_2d = pose_list_2d_orig + pose_list_2d_reproj

    return joined_list_2d


def combine_images(images=[], equal_size=False):
    max_width = 0  # find the max width of all the images
    total_height = 0  # the total height of the images (vertical stacking)

    if equal_size:
        # Find "smaller" image
        min_num_pixels = np.infty
        min_view = None
        for i, img in enumerate(images):
            height = img.shape[0]
            width = img.shape[1]
            num_pixels = height * width
            if num_pixels < min_num_pixels:
                min_num_pixels = num_pixels
                min_view = i

        new_h = images[min_view].shape[0]
        new_w = images[min_view].shape[1]

        # Reshape each image to smallest
        for i, img in enumerate(images):
            if img.shape[0] != new_h or img.shape[1] != new_w:
                img = cv2.resize(img, (new_w, new_h))
                images[i] = img

    for img in images:
        # open all images and find their sizes
        if img.shape[1] > max_width:
            max_width = img.shape[1]
        total_height += img.shape[0]

    # create a new array with a size large enough to contain all the images
    final_image = np.zeros((total_height, max_width, 3), dtype=np.uint8)

    current_y = (
        0  # keep track of where your current image was last placed in the y coordinate
    )
    for image in images:
        # add an image to the final array and increment the y coordinate
        final_image[current_y : image.shape[0] + current_y, : image.shape[1], :] = image
        current_y += image.shape[0]

    return final_image
