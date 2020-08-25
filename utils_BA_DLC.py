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

# pickle utils
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
def load_object(filename):
    with open(filename, 'rb') as input: # note rb and not wb
        return pickle.load(input)
    
def pts_array_from_dlc_3d(dlc_mat):
    ''''input: np.array [n_frames, n_body_parts X n_coords]
    output: np.array [n_frames X n_body_parts, n_coords]'''
    n_frames = dlc_mat.shape[0]
    n_pts_per_frame = int(dlc_mat.shape[1]/3) # for 3d, each point gets 3 cols (x,y,z)
    pts_array = np.zeros((n_frames*n_pts_per_frame, 3)) # 2 cols for x,y coords.
    counter = 0
    for i in np.arange(n_pts_per_frame)*3:
        #print(i)
        pts_array[counter*dlc_mat.shape[0]:\
                  (counter+1)*dlc_mat.shape[0],:] = dlc_mat[:, [i,i+1, i+2]]
        counter += 1
    return pts_array

def inverse_pts_to_dlc3d(pts_array_3d):
    n_frames = int(dlc_mat.shape[0]/3)
    n_pts_per_frame = dlc_mat.shape[1] # for 3d, each point gets 3 cols (x,y,z)
    dlc_mat = np.zeros((n_frames*n_pts_per_frame, 3)) # 2 cols for x,y coords.
    counter = 0
    for i in np.arange(n_pts_per_frame)*3:
        #print(i)
        pts_array[counter*dlc_mat.shape[0]:\
                  (counter+1)*dlc_mat.shape[0],:] = dlc_mat[:, [i,i+1, i+2]]
        counter += 1
    return pts_array