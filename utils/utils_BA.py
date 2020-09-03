#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:47:44 2020

@author: danbiderman
"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as R


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
