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

def pts_array_from_dlc(dlc_mat):
    '''same as above but for 2d arrays.'''
    n_frames = dlc_mat.shape[0]
    n_pts_per_frame = int(dlc_mat.shape[1]/3) # for 2d, each point gets 3 cols (x,y,log_prob)
    pts_array = np.zeros((n_frames*n_pts_per_frame, 2)) # 2 cols for x,y coords.
    counter = 0
    for i in np.arange(n_pts_per_frame)*3:
        #print(i)
        pts_array[counter*dlc_mat.shape[0]:\
                  (counter+1)*dlc_mat.shape[0],:] = dlc_mat[:, [i,i+1]]
        counter += 1
    return pts_array

def chop_dict(dict_to_chop, is3d, start_frame, n_frames):
    
    if is3d:
        key_names = ['x_coords', 'y_coords', 'z_coords']
    else:
        key_names = ['x_coords', 'y_coords']
        
    temp_dict = {}
    for i in range(len(key_names)):
        temp_dict[key_names[i]] = dict_to_chop[key_names[i]][
        :,start_frame:(start_frame+n_frames)]
    
    return temp_dict

def dict_to_arr(dict_3d):
    'convert dicts (in the form above) to arrays. '
    array_temp = np.hstack([dict_3d['x_coords'].T, 
          dict_3d['y_coords'].T,
          dict_3d['z_coords'].T])
    return  array_temp[:, [0,3,6, 1,4,7, 2, 5, 8]]

def arr3d_to_dict(arr):
    'convert n_frames by 9 array to a dict with x,y,z coords'
    dict_3d = {}
    dict_3d["x_coords"] = arr[:,[0, 3, 6]].T
    dict_3d["y_coords"] = arr[:,[1, 4, 7]].T
    dict_3d["z_coords"] = arr[:,[2, 5, 8]].T
    return dict_3d

def revert_ordered_arr_2d_to_dict(body_parts, n_cams, ordered_arr):
    '''remove from here. in IO'''
    n_frames = int(ordered_arr.shape[0]/(body_parts*n_cams))
    coord_list_of_dicts = []
    
    for cam in range(n_cams):
        coord_dict = {}
        coord_dict["x_coords"] = np.zeros((body_parts, n_frames))
        coord_dict["y_coords"] = np.zeros((body_parts, n_frames))
                   
        curr_cam_arr = ordered_arr[cam*n_frames*body_parts:\
                                   (cam+1)*n_frames*body_parts]
#         print(cam*n_frames*body_parts)
#         print( (cam+1)*n_frames*body_parts)

        for f_ind in range(body_parts):
            coord_dict["x_coords"][f_ind, :] = curr_cam_arr[f_ind*n_frames:\
                                                           (f_ind+1)*n_frames, 0]
            coord_dict["y_coords"][f_ind, :] = curr_cam_arr[f_ind*n_frames:\
                                                           (f_ind+1)*n_frames, 1]
        coord_list_of_dicts.append(coord_dict)
    return coord_list_of_dicts

def revert_ordered_arr_to_dict(body_parts, ordered_arr):
    n_frames = int(ordered_arr.shape[0]/body_parts)
    coord_dict = {}
    coord_dict["x_coords"] = np.zeros((body_parts, n_frames))
    coord_dict["y_coords"] = np.zeros((body_parts, n_frames))
    coord_dict["z_coords"] = np.zeros((body_parts, n_frames))
    for f_ind in range(body_parts):
        coord_dict["x_coords"][f_ind, :] = ordered_arr[f_ind*n_frames:\
                                                       (f_ind+1)*n_frames, 0]
        coord_dict["y_coords"][f_ind, :] = ordered_arr[f_ind*n_frames:\
                                                       (f_ind+1)*n_frames, 1]
        coord_dict["z_coords"][f_ind, :] = ordered_arr[f_ind*n_frames:\
                                                       (f_ind+1)*n_frames, 2]
    return coord_dict


