#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 20:06:40 2021

@author: danbiderman

a function that takes in a numpy array with 3D points, 
and transforms it into a pandas array with the bodypart names and (x,y,z) coords.
"""

import numpy as np  # used for reshaping points_3d
import pandas as pd


def points3d_arr_to_df(points_3d, bp_name_list):
    """
    Args:
        points_3d: np.array with shape (num_frames, num_bodyparts, 3)
        bp_name_list: list of strings with len num_bodyparts
    Returns:
        df: a hierarchical pandas data frame with level 1: bp_names and level 2 coords
        of shape (num_frames, num_bodyparts*3)
    """
    points_3d_reshaped = points_3d.reshape(
        points_3d.shape[0], points_3d.shape[1] * points_3d.shape[2]
    )
    coord_names = ["x", "y", "z"]
    hierarchical_index = pd.MultiIndex.from_product(
        [bp_name_list, coord_names], names=["bodyparts", "coords"]
    )
    df = pd.DataFrame(data=points_3d_reshaped, columns=hierarchical_index)
    return df
