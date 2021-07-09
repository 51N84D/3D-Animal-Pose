#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 11:29:37 2021

@author: danbiderman
""" 
import numpy as np 
import scipy
import matplotlib.pyplot as plt
from pathlib import Path
import os

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

def make_meshgrid(mean, grid_dev, num_points=500, allow_pred_outside=True, im_size=None):
    #out = np.expand_dims(np.mgrid[mean[0]-grid_dev:mean[0]+grid_dev:1., \
    #     mean[1]-grid_dev:mean[1]+grid_dev:1.].transpose(1, 2, 0), -1)
    #return out
    if allow_pred_outside:
        x = np.linspace(mean[0]-grid_dev, mean[0]+grid_dev,num_points)
        y = np.linspace(mean[1]-grid_dev,mean[1]+grid_dev,num_points)
    else: # do not cross zero and max values.
        x = np.linspace(np.maximum(mean[0] - grid_dev,0), np.minimum(mean[0] + grid_dev, im_size[1]), num_points)
        y = np.linspace(np.maximum(mean[1] - grid_dev,0), np.minimum(mean[1] + grid_dev, im_size[0]), num_points)
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

def make_arr_for_pca(pts_arr_2d):
    '''
    Args: pts_arr_2d: (num views, num frames, num bodyparts, 2)
    Returns: arr_for_pca (num views * 2, non-nan(num frames * num bodyparts))'''
    pts_2d = pts_arr_2d.reshape(
        pts_arr_2d.shape[0],
        pts_arr_2d.shape[1] * pts_arr_2d.shape[2],
        pts_arr_2d.shape[3],
    )
    pts_2d_filtered, clean_point_indices = clean_nans(pts_2d, True)
    arr_for_pca = np.concatenate((pts_2d_filtered[:,:,0],
                            pts_2d_filtered[:,:,1]),
                             axis=0) # squeeze (x,y) coords with views
    return arr_for_pca

def set_or_open_folder(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print("Opened a new folder at: {}".format(folder_path))
    else:
        print("The folder already exists at: {}".format(folder_path))
    return Path(folder_path) # a PosixPath object


def infer_and_predict(LG_model, arr_squeezed):
    '''Args:
    LG_model: LinearGaussianModel instance.
    arr_squeezed: np.shape(num_features, num_data_points)
    Returns: posterior dict, predictions dict.
    Currently loops over datapoints, compute posterior, and predict.
    the LG_model class could be easily converted to batched ops using @
'''
    # save room
    posteriors = {}
    posteriors["mean"] = np.zeros((LG_model.prior_mean.shape[0], arr_squeezed.shape[1]))
    posteriors["cov"] = np.zeros((LG_model.prior_mean.shape[0], LG_model.prior_mean.shape[0], arr_squeezed.shape[1]))
    predictions = {}
    predictions["mean"] = np.zeros_like(arr_squeezed)
    predictions["cov"] = np.zeros((arr_squeezed.shape[0],
                                   arr_squeezed.shape[0],
                                   arr_squeezed.shape[1]))

    for i in range(arr_squeezed.shape[1]):
        # compute posterior
        post_mean, post_cov = LG_model.compute_posterior(arr_squeezed[:, i])  # a single frame and body part
        posteriors["mean"][:, i], posteriors["cov"][:, :, i] = post_mean.squeeze(), post_cov
        # predict
        mean, cov = LG_model.predict(post_mean, post_cov, False)
        predictions["mean"][:, i], predictions["cov"][:, :, i] = mean.squeeze(), cov

    return posteriors, predictions

def reshape_posts_or_preds(predictions, num_features, num_frames, num_bodyparts):
    """break down num_frames and num_bodyparts"""
    out_dict = {}
    out_dict["mean"] = predictions["mean"].reshape(num_features, num_frames, num_bodyparts)
    out_dict["cov"] = predictions["cov"].reshape(num_features,
                                                num_features,
                                                num_frames,
                                                num_bodyparts)
    return out_dict

def flag_missing_obs(arr_squeezed):
    '''TODO: should be arr_squeezed, or any arr ok?'''
    counts = np.arange(0,np.shape(arr_squeezed)[0] + 2, 2) # assuming if you miss an x coord, you'll miss its y pair
    missing_obs_bool = np.zeros((len(counts), np.shape(arr_squeezed)[-1]), dtype = bool)
    for i in range(len(counts)):
        missing_obs_bool[i,:] = np.sum(np.isnan(arr_squeezed), axis=0)==counts[i]
    assert((np.sum(missing_obs_bool, axis=0)==1).all())
    return missing_obs_bool, counts


def compute_per_view_empirical_maha(LG_model, predictions, arr_squeezed):
    '''loop over views and (bodyparts X frames). compute mahalanobis distance between observation and prediction
    TODO: long with too many loops.'''
    ind_list = create_ind_list(int(arr_squeezed.shape[0] / 2))
    per_view_pred_covs_list = []
    per_view_posterior_mean = []
    per_view_obs = []
    # separate pred_mean, pred_cov, and obs into the different views.
    for i in range(len(ind_list)):
        per_view_pred_covs_list.append([])
        per_view_posterior_mean.append([])
        per_view_obs.append([])
        for j in range(predictions["cov"].shape[-1]):
            per_view_pred_covs_list[i].append(
                LG_model.extract_blocks_from_inds(ind_list[i], predictions["cov"][:, :, j]))
            per_view_posterior_mean[i].append(predictions["mean"][ind_list[i], j])
            per_view_obs[i].append(arr_squeezed[ind_list[i], j])

    # compute maha distance
    maha_list = []
    for i in range(len(ind_list)):
        maha_list.append([])
        for j in range(predictions["cov"].shape[-1]):
            diff = per_view_obs[i][j] - per_view_posterior_mean[i][j]
            maha_list[i].append(diff.T @ np.linalg.inv(per_view_pred_covs_list[i][j]) @ diff)

    return maha_list

def compute_percentile_in_list(maha_list, percentile=95):
    empirical_ds = np.zeros(len(maha_list))
    for i in range(len(maha_list)):
        empirical_ds[i] = np.nanpercentile(maha_list[i], percentile)
    return empirical_ds


