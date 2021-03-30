#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 11:29:37 2021

@author: danbiderman
""" 
import numpy as np 
import scipy
import matplotlib.pyplot as plt

def mahalanobis_distance(cov, mean, stacked_meshgrid):
    '''Args: stacked_meshgrid: shape(,,2,1)
        cov: shape(2,2)
        mean: shape(2,1)
        Output: '''
    assert(stacked_meshgrid.shape[-2:] == (2,1))
    mean_subtracted = stacked_meshgrid - mean
    out = mean_subtracted.transpose(0, 1, 3, 2)@ \
    np.linalg.inv(cov)@mean_subtracted
    return out.reshape(out.shape[0], out.shape[1])

## test
#cov_test = np.random.normal(size = (2,2))
#mean_test = np.random.normal(size = (2,1))
#stacked_meshgrid = np.random.normal(size = (100, 1000, 2,1))
#out = mahalanobis_distance(cov_test, mean_test, stacked_meshgrid)
#out.shape

def make_meshgrid(mean, grid_dev):
    #out = np.expand_dims(np.mgrid[mean[0]-grid_dev:mean[0]+grid_dev:1., \
    #     mean[1]-grid_dev:mean[1]+grid_dev:1.].transpose(1, 2, 0), -1)
    #return out
    x = np.linspace(mean[0]-grid_dev,mean[0]+grid_dev,300)
    y = np.linspace(mean[1]-grid_dev,mean[1]+grid_dev,300)
    x,y = np.meshgrid(x,y)
    stacked_meshgrid = np.expand_dims(np.stack([x,y], axis=-1), 
                                      axis=-1)
    return x,y,stacked_meshgrid
## test
#x,y,stacked = make_meshgrid(mean, 100)
#print(stacked.shape)
#d = mahalanobis_distance(cov=cov, mean=mean, stacked_meshgrid=stacked)
#d.shape
#plt.contour(x,y,d,[3.0])

def plot_conf_ellipse(Sigma, mean, confidence, color, n):
    '''this approach is using the chi2 confidence interval of the MV normal'''
    from scipy.stats import multivariate_normal,chi2
    from scipy.linalg import sqrtm, inv
    radius = np.sqrt( chi2.isf(1-confidence, df=n) )
    T = np.linspace(0, 2*np.pi, num=1000)
    circle = radius * np.vstack([np.cos(T), np.sin(T)])
    x, y = sqrtm(Sigma) @ circle
    plt.plot(x+mean[0], y+mean[1], color=color, linewidth=1, alpha=.5)

def undo_nan_cleanup(original_arr, clean_point_indices, cleaned_arr):
    # ToDo: add a condition that acounts for partial nan removal
    all_point_indices = np.arange(original_arr.shape[1])
    
    nan_point_indices = np.asarray(
        [x for x in all_point_indices if x not in clean_point_indices]
    )
    filled_points_2d = np.empty(original_arr.shape)
    filled_points_2d[:, clean_point_indices, :] = cleaned_arr
    filled_points_2d[:, nan_point_indices, :] = np.nan
    return filled_points_2d
# nan_undone = undo_nan_cleanup(pts_2d, 
#                               clean_point_indices, 
#                               pts_2d_filtered)

def extract_missing_ind_pairs(obs_vec):
    '''assuming obs_vec = [K*X, K*Y] coords where K==num_views'''
    inds_to_show = np.where(np.isnan(obs_vec))[0]
    num_views = int(len(obs_vec)/2)
    mods = inds_to_show % num_views
    unique_mods = np.unique(mods)
    ind_pairs = []
    for i in unique_mods:
        ind_pairs.append(inds_to_show[mods == i])
    return ind_pairs

# TODO: make sure it's not duplicated in utils.utils_BA
def clean_nans(pts_array_2d=None, drop_any_nan_row=False):
    # pts_array_2d should be (num_cams, num_frames * num_bodyparts, 2)
    # Clean up nans
    count_nans = np.sum(np.isnan(pts_array_2d), axis=0)[:, 0]
    if drop_any_nan_row:
        nan_rows = count_nans>0
    else:
        nan_rows = count_nans > pts_array_2d.shape[0] - 2
    pts_all_flat = np.arange(pts_array_2d.shape[1])
    pts_2d_filtered = pts_array_2d[:, ~nan_rows, :]
    clean_point_indices = pts_all_flat[~nan_rows]
    return pts_2d_filtered, clean_point_indices

# this applies to Sawtell's preprocessing in general.
def extract_bp_idx(skeleton_names, 
                  view_names,
                  bp_to_keep):
    
    multiview_idx_to_name = {}
    multiview_name_to_idx = {}
    
    for view_name in view_names:
        multiview_name_to_idx[view_name] = []
        new_skeleton_names = []
    
    for idx, name in enumerate(skeleton_names):  # was prev f["skeleton_names"] building on the labels data.
        if len(bp_to_keep) > 0:
            skip_bp = True
            for bp in bp_to_keep:
                if (
                    bp == name.split("_")[0]
                ):  # bp == name.decode("UTF-8").split("_")[0]:
                    skip_bp = False
            if skip_bp:
                continue

        new_skeleton_names.append(name)
        for view_name in view_names:
            if view_name in name.split("_")[-1]:  # name.decode("UTF-8").split("_")[-1]:
                multiview_idx_to_name[idx] = view_name
                multiview_name_to_idx[view_name].append(idx)
    
    return multiview_idx_to_name, multiview_name_to_idx, new_skeleton_names

# extract parameters of interest for linear gaussian model 
def get_PCA_params(pca, num_PCs, arr_for_pca, parametrization = "Bishop"):
    cov = pca.get_covariance() # C: (2K \times 2K)
    components = pca.components_[:num_PCs,:] # P: (M \times 2K), our A 
    mu = np.mean(arr_for_pca, axis=1) # b: 2K 
    e_vals = pca.explained_variance_
    D = np.diag(e_vals[:num_PCs]) # D: M \times M diagonal matrix with top M e-vals as entries
    sigma_2 = np.mean(e_vals[num_PCs:]) # mean of discarded e-vals
    PDP_TOP = np.linalg.multi_dot([components.T, D, components]) # cov mat prediction by top M evals
    R = cov - PDP_TOP
    param_dict = {}
    param_dict["prior_mean"] = np.zeros((num_PCs,1))
    param_dict["obs_offset"] = mu

    if parametrization == "LP":
        param_dict["prior_precision"] = np.linalg.inv(D)
        param_dict["obs_precision"] = np.linalg.inv(R+np.eye(R.shape[0])*0.001)
        param_dict["obs_projection"] = components.T
    elif parametrization == "Bishop":
        param_dict["prior_precision"] = np.eye(num_PCs)
        param_dict["obs_precision"] = np.linalg.inv(np.eye(len(mu))*sigma_2)
        param_dict["obs_projection"] = np.dot(components.T, \
                                              scipy.linalg.sqrtm(D-np.eye(num_PCs)*sigma_2))
    
    return param_dict

def plot_coords_views(true_pts, pred_pts, title_str):
    '''across all bodyparts.'''
    fig, ax = plt.subplots(pred_pts.shape[0],
                      pred_pts.shape[-1])
    for i in range(pred_pts.shape[0]):
        for j in range(pred_pts.shape[-1]):
            ax[i,j].plot(true_pts[i,:,j].T, 'r')
            ax[i,j].plot(pred_pts[i,:,j].T, 'k--')
            ax[i,j].set_title('view: %i coord: %i' % (i,j))
    ax[i,j].legend(['true', 'pred'])
    fig.suptitle(title_str)
    fig.tight_layout()
    return fig

def create_ind_list(num_views):
    """function for getting a list of inds. we use these inds to select elements
    of covariance matrices and means. it assumes that all x coords, followed by all y coords"""
    ind_list = []
    for i in range(num_views):
        ind_list.append(np.array([i, i+num_views]))
    return ind_list