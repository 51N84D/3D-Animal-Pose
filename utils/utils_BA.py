#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:47:44 2020

@author: danbiderman
"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as R
from copy import deepcopy

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid="ignore"):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return (
        cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v
    )


# Nonlinear projection
def project_og(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj ** 2, axis=1)
    r = 1 + k1 * n + k2 * n ** 2
    points_proj *= (r * f)[:, np.newaxis]

    return points_proj


# Linear projection
def project(points, camera_params, offset):
    """Convert 3-D points to 2-D by projecting onto images."""
    # -----Get extrinsics------
    rot_vec = R.from_rotvec(camera_params[:, :3])
    rot_mat = rot_vec.as_matrix()
    t = camera_params[:, 3:6]

    # ----Get intrinsics----
    f = camera_params[:, 6]
    p_x = offset[:, 0]
    p_y = offset[:, 1]

    K = np.zeros((f.shape[0], 3, 3))
    indices = np.arange(f.shape[0])

    # Let skew stay 0
    K[indices, 0, 0] = f[indices]
    K[indices, 1, 1] = f[indices]
    K[indices, 0, 2] = p_x[indices]
    K[indices, 1, 2] = p_y[indices]
    K[indices, 2, 2] = 1

    P = np.matmul(K, np.concatenate((rot_mat, t[:, :, np.newaxis]), axis=2))

    ones = np.ones((points.shape[0], 1))
    homog_points = np.concatenate((points, ones), axis=1)
    points_proj = np.squeeze(np.matmul(P, homog_points[:, :, np.newaxis]))
    points_proj = points_proj[:, 0:2]
    return points_proj


# Linear projection instead of nonlinear w/ distortion


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d, offset):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[: n_cameras * 7].reshape((n_cameras, 7))
    points_3d = params[n_cameras * 7 :].reshape((n_points, 3))
    points_proj = project(
        points_3d[point_indices], camera_params[camera_indices], offset[camera_indices]
    )
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 7 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(7):
        A[2 * i, camera_indices * 7 + s] = 1
        A[2 * i + 1, camera_indices * 7 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 7 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 7 + point_indices * 3 + s] = 1

    return A


def filter_points(points_2d, num_views=2):
    # Keep only points that appear in >= num_views
    # Shape can be either (num_views, num_frames, num_bodyparts, 2) or
    # (num_views, num_frames  * num_bodparts, 2)
    assert len(points_2d.shape) in (3, 4)
    points_2d_new = deepcopy(points_2d)
    if len(points_2d.shape) == 4:
        nan_counts = np.sum(np.any(np.isnan(points_2d), axis=-1), axis=0)
        indices_to_nan = nan_counts < num_views
        points_2d_new[:, indices_to_nan, :] = np.nan
    return points_2d_new
    # return np.all(np.any(np.isnan(points_2d_og), axis=-1), axis=0)


def clean_nans(pts_array_2d):
    # pts_array_2d should be (num_cams, num_frames * num_bodyparts, 2)
    # Clean up nans
    count_nans = np.sum(np.isnan(pts_array_2d), axis=0)[:, 0]
    nan_rows = count_nans > pts_array_2d.shape[0] - 2

    pts_all_flat = np.arange(pts_array_2d.shape[1])
    pts_2d_filtered = pts_array_2d[:, ~nan_rows, :]
    clean_point_indices = pts_all_flat[~nan_rows]
    return pts_2d_filtered, clean_point_indices


def refill_arr(points_2d, info_dict):
    points_2d = deepcopy(points_2d)
    clean_point_indices = info_dict["clean_point_indices"]
    num_points_all = info_dict["num_points_all"]
    num_frames = info_dict["num_frames"]
    all_point_indices = np.arange(num_points_all)
    print(clean_point_indices)
    print(points_2d.shape)

    nan_point_indices = np.asarray(
        [x for x in all_point_indices if x not in clean_point_indices]
    )
    print(nan_point_indices)

    # If points_2d is (num_views, num_points, 2):
    if len(points_2d.shape) == 3:
        if len(nan_point_indices) == 0:
            return points_2d.reshape(
                points_2d.shape[0], num_frames, -1, points_2d.shape[-1]
            )

        filled_points_2d = np.empty((points_2d.shape[0], num_points_all, 2))
        filled_points_2d[:, clean_point_indices, :] = points_2d
        filled_points_2d[:, nan_point_indices, :] = np.nan
        filled_points_2d = np.reshape(
            filled_points_2d, (points_2d.shape[0], num_frames, -1, 2)
        )

    # Elif points2d is (num_points, 2) --> this happens if we consider each view separately
    elif len(points_2d.shape) == 2:
        if len(nan_point_indices) == 0:
            return points_2d.reshape(num_frames, -1, points_2d.shape[-1])

        filled_points_2d = np.empty((num_points_all, 2))
        filled_points_2d[clean_point_indices, :] = points_2d
        filled_points_2d[nan_point_indices, :] = np.nan
        filled_points_2d = np.reshape(filled_points_2d, (num_frames, -1, 2))

    return filled_points_2d