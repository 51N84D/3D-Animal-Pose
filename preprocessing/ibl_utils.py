"""Helper functions to preprocess IBL data from raw data to HDF5 files for behavenet."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

#sys.path.append('/home/mattw/Dropbox/github/3D-Animal-Pose')
#from utils.utils_IO import ordered_arr_3d_to_dict, refill_nan_array
#from anipose_BA import CameraGroup, Camera


# -------------------------------------------------------------------------------------------------
# Manipulate DLC makers
# -------------------------------------------------------------------------------------------------

def get_markers(alf_path, view, likelihood_thresh=0.9):
    """Load DLC markers and likelihood masks from alf directory.

    Parameters
    ----------
    alf_path : str
        path to alf directory that contains dlc markers
    view : str
        'left' | 'right' | 'body'
    likelihood_thresh : float
        dlc likelihoods below this value returned as NaNs

    Returns
    -------
    tuple
        - XYs (dict): keys are body parts, values are np.ndarrays of shape (n_t, 2)
        - masks (dict): keys are body parts, values are np.ndarrays of shape (n_t, 2)

    """

    import pandas as pd

    dlc_path = os.path.join(alf_path, '_ibl_%sCamera.dlc.pqt' % view)
    cam = pd.read_parquet(dlc_path)

    points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])
    if view != 'body':
        d = list(points)
        d.remove('tube_top')
        d.remove('tube_bottom')
        points = np.array(d)

    # Set values to nan if likelyhood is too low # for pqt: .to_numpy()
    XYs = {}
    masks = {}
    likelihoods = {}
    for point in points:
        x = np.ma.masked_where(
            cam[point + '_likelihood'].to_numpy() < likelihood_thresh,
            cam[point + '_x'].to_numpy())
        x = x.filled(np.nan)

        y = np.ma.masked_where(
            cam[point + '_likelihood'].to_numpy() < likelihood_thresh,
            cam[point + '_y'].to_numpy())
        y = y.filled(np.nan)
        
        XYs[point] = np.hstack([x[:, None], y[:, None]])
        masks[point] = np.ones_like(XYs[point])
        masks[point][np.isnan(XYs[point])] = 0
        likelihoods[point] = cam[point + '_likelihood'].to_numpy()

    return XYs, masks, likelihoods


def get_tail_position(markers):
    """Find median x/y position of tail base in body videos.

    Parameters
    ----------
    markers : dict
        keys are body parts, values are np.ndarrays of shape (n_t, 2); must contain `'tail_start'`

    Returns
    -------
    tuple
        - x-value (float)
        - y-value (float)

    """
    tail_x = markers['tail_start'][:, 0]
    tail_y = markers['tail_start'][:, 1]
    median_x = np.nanmedian(tail_x)
    median_y = np.nanmedian(tail_y)
    return median_x, median_y


def get_pupil_position(markers):
    """Find median x/y position of pupil in left/right videos.

    Parameters
    ----------
    markers : dict
        keys are body parts, values are np.ndarrays of shape (n_t, 2); must contain
        `'pupil_bottom_r'`, `'pupil_left_r'`, `'pupil_right_r'`, `'pupil_top_r'` or equivalent for
        left side

    Returns
    -------
    tuple
        - x-value (float)
        - y-value (float)

    """
    if 'pupil_bottom_r' in list(markers.keys()):
        pupil_markers = ['pupil_bottom_r', 'pupil_left_r', 'pupil_right_r', 'pupil_top_r']
    else:
        pupil_markers = ['pupil_bottom_l', 'pupil_left_l', 'pupil_right_l', 'pupil_top_l']
    pupil_x = []
    pupil_y = []
    for pm in pupil_markers:
        pupil_x.append(markers[pm][:, 0, None])
        pupil_y.append(markers[pm][:, 1, None])
    pupil_x = np.hstack(pupil_x)
    pupil_y = np.hstack(pupil_y)
    median_x = np.nanmedian(pupil_x)
    median_y = np.nanmedian(pupil_y)
    return median_x, median_y


def get_nose_position(markers):
    """Find median x/y position of nose tip in left/right videos.

    Parameters
    ----------
    markers : dict
        keys are body parts, values are np.ndarrays of shape (n_t, 2); must contain `'nose_tip'`

    Returns
    -------
    tuple
        - x-value (float)
        - y-value (float)

    """
    return np.nanmedian(markers['nose_tip'], axis=0)


def get_subsampled_markers(markers, view, body_part, idx_aligned=None):
    """Subsample markers in either space or time.

    For the left view, divide coordinates by 2 two get same (lower) spatial resolution of right cam
    For the right vew, subsample timepoints based on provided indices to get same (lower) temporal
    resolution of left cam

    Parameters
    ----------
    markers : dict
        keys are camera views, values are dicts
        for these dicts, keys are body parts, values are np.ndarrays of shape (n_t, 2)
    view : str
        'left' | 'right' | 'body'
    body_part : str
        name of dlc marker, e.g. 'paw_l', 'paw_r', 'nose_tip'
    idx_aligned : array-like, optional
        set of time indices to subsample markers from right view

    Returns
    -------
    array-like
        subsampled markers for specified body part

    """
    if view == 'right':
        # fast camera; subsample in time using alignment indices
        return markers[view][body_part][idx_aligned]
    elif view == 'left':
        # slow camera; subsample in space (divide by half)
        return markers[view][body_part] / 2
    else:
        raise Exception


def align_markers_from_different_views(markers, marker_names, idx_aligned):
    """Align 2d DLC markers from multiple views into common spatiotemporal format.

    Note: left camera has twice the resolution of the right camera, left cam has 60 Hz sampling
    rate, right cam has 150 Hz sampling rate

    Parameters
    ----------
    markers : dict
        keys are camera views, values are dicts
        for these dicts, keys are body parts, values are np.ndarrays of shape (n_t, 2)
    marker_names : list
        list of marker names to align, e.g. ['paw_l', 'paw_r', 'nose_tip']
    idx_aligned : array-like
        set of time indices to subsample markers from right view

    Returns
    -------
    tuple
        - 2d points array (np.ndarray): shape of (n_cameras, n_t * n_markers, 2) where final "2" is
          for x, y; this is the format needed for running bundle adjustment code
        - info dict (dict): contains info about frames, cameras, nan indices, etc.

    """
    # paw_l in video left = paw_r in video right
    # Divide left coordinates by 2 to get them in half resolution like right cam;
    # reduce temporal resolution of right cam to that of left cam
    num_analyzed_body_parts = len(marker_names)

    cam_right_paw1 = get_subsampled_markers(markers, 'right', 'paw_l', idx_aligned)
    cam_left_paw1 = get_subsampled_markers(markers, 'left', 'paw_r')

    cam_right_paw2 = get_subsampled_markers(markers, 'right', 'paw_r', idx_aligned)
    cam_left_paw2 = get_subsampled_markers(markers, 'left', 'paw_l')

    cam_right_nose = get_subsampled_markers(markers, 'right', 'nose_tip', idx_aligned)
    cam_left_nose = get_subsampled_markers(markers, 'left', 'nose_tip')

    # the format shall be such that points are concatenated, p1,p2,p3,p1,p2,p3, ...
    cam_right = np.zeros((len(idx_aligned) * num_analyzed_body_parts, 2))

    # order is paw_r, paw_l, nose_tip
    cam_right[0::3] = cam_right_paw1
    cam_right[1::3] = cam_right_paw2
    cam_right[2::3] = cam_right_nose

    cam_left = np.zeros((len(idx_aligned) * num_analyzed_body_parts, 2))
    cam_left[0::3] = cam_left_paw1
    cam_left[1::3] = cam_left_paw2
    cam_left[2::3] = cam_left_nose

    pts_array_2d_with_nans = np.array([cam_right, cam_left])

    num_cameras, num_points_all, _ = pts_array_2d_with_nans.shape

    # remove nans (any of the x_r,y_r, x_l, y_l) and keep clean_point_indices
    non_nan_idc = ~np.isnan(pts_array_2d_with_nans).any(axis=2).any(axis=0)

    info_dict = {}
    info_dict['num_frames'] = cam_left_paw1.shape[0]
    info_dict['num_cameras'] = num_cameras
    info_dict['num_analyzed_body_parts'] = num_analyzed_body_parts
    info_dict['num_points_all'] = num_points_all
    info_dict['clean_point_indices'] = np.arange(num_points_all)[non_nan_idc]

    pts_array_2d = pts_array_2d_with_nans[:, info_dict['clean_point_indices']]

    # pts_array_2d: (n_camera, n_t * n_markers, 2) where final 2 is x, y
    return pts_array_2d, info_dict


def infer_3d_markers(
        pts_array_2d, info_dict, marker_names, img_width, img_height, focal_length_mm,
        sensor_size):
    """Infer 3d trajectories using two or more camera views with the bundle adjustment algorithm.

    Parameters
    ----------
    pts_array_2d : np.ndarray
        shape of (n_cameras, n_t * n_markers, 2) where final "2" is for x, y; this is the format
        needed for running bundle adjustment code
    info_dict : dict
        contains info about frames, cameras, nan indices, etc.; output from
        `align_markers_from_different_views`
    marker_names : list
        e.g. ['paw_l', 'paw_r', 'nose_tip']
    img_width : int
        shared width of images in pixels
    img_height : int
        shared height of images in pixels
    focal_length_mm : float
        shared focal length of cameras in mm
    sensor_size : float
        size of sensor...

    Returns
    -------
    tuple
        - 3d points w/ nans (dict): keys are body parts, values are np.ndarrays of shape
          (n_time, 3) where the 3 is x, y, z coords
        - 3d points w/o nans (np.ndarray): shape (n_time * n_markers, 3); this is the format needed
          to perform reprojections into the 2d camera coordinates
        - likelihoods (dict): keys are body parts, values are arrays of shape (n_time,); only a
          single likelihood is used for all three coordinates; will be 1 for present markers, 0 for
          absent markers
        - cameras (dict): keys are views, values are Camera objects from anipose package; used to
          perform reprojectoins into the 2d camera coordinates

    """

    P_X_left = P_X_right = img_width // 2
    P_Y_left = P_Y_right = img_height // 2

    focal_length_1 = (focal_length_mm * img_width) / sensor_size
    focal_length_2 = (focal_length_mm * img_width) / sensor_size

    # For IBL we give 3D coordinates in resolution of right camera (camera 1)
    # Thus offsets are the same, however the 2D points of the left cam must have been divided
    # by 2

    # --------INIT CAMERA 1------------ (that's ibl_rightCamera - called TOP)
    # right-handed coordinate system, e_z how cam 1 points;
    # e_x perp to plane with both cams and target
    camera_1 = Camera(rvec=[0, 0, 0], tvec=[0, 0, 0])
    # Set offset
    camera_1.set_size((P_X_right, P_Y_right))

    # initialize camera params
    cam1_init_params = np.abs(np.random.rand(8))
    # Set rotations [0:3] and translation [3:6] to 0
    cam1_init_params[0:6] = 0
    # Initialize focal length to image width
    cam1_init_params[6] = focal_length_1
    # Initialize distortion to 0
    cam1_init_params[7] = 0.0
    # Set parameters
    camera_1_mat = camera_1.get_camera_matrix()
    camera_1_mat[0, 2] = P_X_right
    camera_1_mat[1, 2] = P_Y_right
    camera_1.set_camera_matrix(camera_1_mat)
    camera_1.set_params(cam1_init_params)

    # --------INIT CAMERA 2------------(that's ibl_leftCamera)
    # Set rotation vector w.r.t. camera 1
    # roration around y axis only, about 120 deg (2.0127 rad) from Guido's CAD
    rvec2 = np.array([0, 2.0127, 0])
    # Set translation vector w.r.t. camera 1, using CAD drawing [mm];
    # cameras are 292.8 mm apart;
    # distance vector pointing from cam1 to the other cam:
    tvec2 = [-1.5664, 0, 2.4738]
    # Initialize camera 2
    camera_2 = Camera(rvec=rvec2, tvec=tvec2)
    # Set offset
    camera_1.set_size((P_X_left, P_Y_left))

    cam2_init_params = np.abs(np.random.rand(8))
    cam2_init_params[0:3] = rvec2
    cam2_init_params[3:6] = tvec2
    cam2_init_params[6] = focal_length_2
    cam2_init_params[7] = 0.0
    camera_2.set_params(cam2_init_params)
    camera_2_mat = camera_2.get_camera_matrix()
    camera_2_mat[0, 2] = P_X_left
    camera_2_mat[1, 2] = P_Y_left
    camera_2.set_camera_matrix(camera_2_mat)

    # Group cameras
    cam_group = CameraGroup(cameras=[camera_1, camera_2])

    # Get error before Bundle Adjustment by triangulating using the initial parameters:
    f0, points_3d_init = cam_group.get_initial_error(pts_array_2d)
    print(points_3d_init.shape)

    print('----CAMERA 1-----')
    print(camera_1.get_camera_matrix())
    print('----CAMERA 2-----')
    print(camera_2.get_camera_matrix())

    # fig = plot_cams_and_points(
    #     cam_group=cam_group, points_3d=points_3d_init, title="3D Points Initialized")

    # Run Bundle Adjustment
    res, points_3d_raw = cam_group.bundle_adjust(pts_array_2d, max_nfev=200)

    # do the pts_array_3d_clean
    array_3d_back = refill_nan_array(points_3d_raw, info_dict, dimension='3d')

    # get 3d points nicely organized
    pts3d_dict = ordered_arr_3d_to_dict(array_3d_back, info_dict)
    points_3d = {}
    for i, m in enumerate(marker_names):
        points_3d[m] = np.array([
            pts3d_dict['x_coords'][:, i],
            pts3d_dict['y_coords'][:, i],
            pts3d_dict['z_coords'][:, i]]).T

    # get corresponding likelihoods nicely organized
    # 0 if only one view has a non-nan value
    # 1 if two or more views have non-nan values
    likelihoods = {}
    for m in marker_names:
        ls = np.ones(points_3d[m].shape[0])
        bad_frames = np.unique(np.where(np.isnan(points_3d[m]))[0])
        ls[bad_frames] = 0
        likelihoods[m] = ls

    return points_3d, points_3d_raw, likelihoods, {'right': camera_1, 'left': camera_2}


def reproject_3d_markers(points_3d, info_dict, cameras, marker_names):
    """Use 3d trajectories and camera info to find 2d pixel values for each camera view.

    Parameters
    ----------
    points_3d : np.ndarray
        shape (n_time * n_markers, 3)
    info_dict : dict
        contains info about frames, cameras, nan indices, etc.; output from
        `align_markers_from_different_views`
    cameras : dict
        keys are views, values are Camera objects from anipose package
    marker_names : list
        e.g. ['paw_l', 'paw_r', 'nose_tip']

    Returns
    -------
    dict
        keys are camera views, values are dicts
        for those dicts, keys are body parts, values are np.ndarrays of shape (n_time, 2)

    """
    n = len(marker_names)
    views = list(cameras.keys())
    points_2d = {}
    for view in views:
        # reproject 3d points into 2d
        points_proj_sub = cameras[view].project(points_3d).squeeze()
        # insert original nans
        points_proj = refill_nan_array_2d(points_proj_sub, info_dict)
        # organize
        points_2d[view] = {m: points_proj[i::n] for i, m in enumerate(marker_names)}
    return points_2d


def crop_markers(markers, xmin, xmax, ymin, ymax):
    """Update marker values to reflect crop of corresponding images.

    Parameters
    ----------
    markers : dict or array-like
        if dict, keys are body parts, values are np.ndarrays of shape (n_time, 2)
        if array-like, np.ndarray of shape (n_time, 2)
    xmin : float
        min x value from image crop
    xmax : float
        max x value from image crop
    ymin : float
        min y value from image crop
    ymax : float
        max y value from image crop

    Returns
    -------
    variable
        same type as input, with updated marker values

    """
    if isinstance(markers, dict):
        marker_names = list(markers.keys())
        markers_crop = {}
        for m in marker_names:
            markers_crop[m] = markers[m] - np.array([xmin, ymin])
    else:
        markers_crop = markers - np.array([xmin, ymin])
    return markers_crop


def scale_markers(markers, xpix_old, xpix_new, ypix_old, ypix_new):
    """Update marker values to reflect scale of corresponding images.

    Parameters
    ----------
    markers : dict or array-like
        if dict, keys are body parts, values are np.ndarrays of shape (n_time, 2)
        if array-like, np.ndarray of shape (n_time, 2)
    xpix_old : int
        xpix of original images
    xpix_new
        xpix of new images
    ypix_old
        ypix of old images
    ypix_new
        ypix of new images

    Returns
    -------
    variable
        same type as input, with updated marker values

    """
    old = np.array([xpix_old, ypix_old])
    new = np.array([xpix_new, ypix_new])
    if isinstance(markers, dict):
        marker_names = list(markers.keys())
        markers_scale = {}
        for m in marker_names:
            markers_scale[m] = (markers[m] / old) * new
    else:
        markers_scale = (markers / old) * new
    return markers_scale


# -------------------------------------------------------------------------------------------------
# video helpers
# -------------------------------------------------------------------------------------------------

def get_frames_from_idxs(cap, idxs):
    """Helper function to load video segments.

    Parameters
    ----------
    cap : cv2.VideoCapture object
    idxs : array-like
        frame indices into video

    Returns
    -------
    np.ndarray
        returned frames of shape shape (n_frames, y_pix, x_pix)

    """
    is_contiguous = np.sum(np.diff(idxs)) == (len(idxs) - 1)
    n_frames = len(idxs)
    for fr, i in enumerate(idxs):
        if fr == 0 or not is_contiguous:
            cap.set(1 , i)
        ret, frame = cap.read()
        if ret:
            if fr == 0:
                height, width, _ = frame.shape
                frames = np.zeros((n_frames, 1, height, width), dtype='uint8')
            frames[fr, 0, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            print(
                'warning! reached end of video; returning blank frames for remainder of ' +
                'requested indices')
            break
    return frames


def get_frame_lims(x_eye, y_eye, x_nose, y_nose, view):
    """

    Parameters
    ----------
    x_eye
    y_eye
    x_nose
    y_nose
    view

    Returns
    -------

    """
    # horizontal proportions
    edge2nose = 0.02
    nose2eye = 0.33
    eye2edge = 0.65
    # vertical proportions
    eye2top = 0.10
    eye2bot = 0.90
    # horizontal calc
    nose2eye_pix = np.abs(x_eye - x_nose)
    edge2nose_pix = edge2nose / nose2eye * nose2eye_pix
    eye2edge_pix = eye2edge / nose2eye * nose2eye_pix
    total_x_pix = np.round(nose2eye_pix + edge2nose_pix + eye2edge_pix)
    if view == 'left':
        xmin = int(x_nose - edge2nose_pix)
        xmax = int(x_eye + eye2edge_pix)
    elif view == 'right':
        xmin = int(x_eye - eye2edge_pix)
        xmax = int(x_nose + edge2nose_pix)
    else:
        raise Exception
    # vertical calc (assume we want a square image out)
    eye2top_pix = eye2top * total_x_pix
    eye2bot_pix = eye2bot * total_x_pix
    ymin = int(y_eye - eye2top_pix)
    ymax = int(y_eye + eye2bot_pix)
    return xmin, xmax, ymin, ymax


def refill_nan_array_2d(pts_array_clean, info_dict):
    """

    Parameters
    ----------
    pts_array_clean
    info_dict

    Returns
    -------

    """
    pts_refill = np.empty(
        (info_dict["num_frames"] * info_dict["num_analyzed_body_parts"], 2))
    pts_refill[:] = np.NaN
    pts_refill[info_dict["clean_point_indices"], :] = pts_array_clean
    return pts_refill


# -------------------------------------------------------------------------------------------------
# Plotting/movies
# -------------------------------------------------------------------------------------------------

def plot_labeled_frames(frames, points_2d_og=None, points_2d_reproj=None, height=7):
    """

    Parameters
    ----------
    frames
    points_2d_og
    points_2d_reproj
    height

    Returns
    -------

    """
    color_reproj = [[1, 0, 1]]
    color_orig = [[1, 1, 0]]

    views = list(frames.keys())
    img_height, img_width = frames[views[0]].shape

    h = height
    w = h * (img_width / img_height)

    fig, axes = plt.subplots(1, 2, figsize=(w, h))
    # fig.tight_layout(pad=0)
    plt.subplots_adjust(wspace=0, hspace=0, bottom=0, left=0, right=1, top=1)

    txt_kwargs = {
        'fontsize': 16, 'color': [1, 1, 1], 'horizontalalignment': 'left',
        'verticalalignment': 'center', 'transform': axes[1].transAxes}

    for i, view in enumerate(views):

        axes[i].imshow(frames[view], vmin=0, vmax=255, cmap='gray')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].text(
            0.95, 0.05, '%s view' % view, fontsize=16, color=[1, 1, 1],
            horizontalalignment='right', verticalalignment='center',
            transform=axes[i].transAxes)

        # reprojected points
        if points_2d_reproj is not None:
            for m in ['paw_l', 'paw_r', 'nose_tip']:
                axes[i].scatter(
                    points_2d_reproj[view][m][0],
                    points_2d_reproj[view][m][1],
                    s=40, c=color_reproj)
        # original points
        if points_2d_og is not None:
            for m in ['paw_l', 'paw_r', 'nose_tip']:
                if view == 'left':
                    axes[i].scatter(
                        points_2d_og[view][m][0],
                        points_2d_og[view][m][1],
                        s=40, c=color_orig)
                elif view == 'right':
                    axes[i].scatter(
                        points_2d_og[view][m][0],
                        points_2d_og[view][m][1],
                        s=40, c=color_orig)

        if i == 1 and (points_2d_og is not None and points_2d_reproj is not None):
            axes[i].scatter(0.05 * img_width, 0.85 * img_height, s=40, c=color_orig)
            axes[i].text(0.08, 0.15, 'Original marker', **txt_kwargs)

            axes[i].scatter(0.05 * img_width, 0.95 * img_height, s=40, c=color_reproj)
            axes[i].text(0.08, 0.05, 'Reprojected marker', **txt_kwargs)

    plt.show()


def make_labeled_movie(
        filename, frames, points_2d_og=None, points_2d_reproj=None, framerate=20, height=4):
    """

    Parameters
    ----------
    filename
    frames
    points_2d_og
    points_2d_reproj
    framerate
    height

    Returns
    -------

    """

    import matplotlib.animation as animation
    from matplotlib.animation import FFMpegWriter

    views = list(frames.keys())
    n_frames = len(frames[views[0]])
    img_height, img_width = frames[views[0]][0].shape

    h = height
    w = 2 * h * (img_width / img_height)
    fig, axes = plt.subplots(1, 2, figsize=(w, h))
    plt.subplots_adjust(wspace=0, hspace=0, left=0, bottom=0, right=1, top=1)
    for ax, view in zip(axes, views):
        ax.set_yticks([])
        ax.set_xticks([])
        ax.text(
            0.95, 0.05, '%s view' % view, fontsize=16, color=[1, 1, 1],
            horizontalalignment='right', verticalalignment='center',
            transform=ax.transAxes)

    color_reproj = [[1, 0, 1]]
    color_orig = [[1, 1, 0]]

    im_kwargs = {'animated': True, 'vmin': 0, 'vmax': 255, 'cmap': 'gray'}
    txt_kwargs = {
        'fontsize': 16, 'color': [1, 1, 1], 'horizontalalignment': 'left',
        'verticalalignment': 'center', 'transform': axes[1].transAxes}

    if points_2d_reproj is not None:
        axes[1].text(0.08, 0.05, 'Reprojected marker', **txt_kwargs)
    if points_2d_og is not None:
        axes[1].text(0.08, 0.15, 'Original marker', **txt_kwargs)

    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = []
    for n in range(n_frames):

        if n % 100 == 0:
            print('processing frame %03i/%03i' % (n, n_frames))

        ims_curr = []

        for i, view in enumerate(views):

            im = axes[i].imshow(frames[view][n], **im_kwargs)
            ims_curr.append(im)

            # reprojected points
            if points_2d_reproj is not None:
                for m in ['paw_l', 'paw_r', 'nose_tip']:
                    im = axes[i].scatter(
                        points_2d_reproj[view][m][n, 0], points_2d_reproj[view][m][n, 1],
                        s=40, c=color_reproj)
                    ims_curr.append(im)
            # original points
            if points_2d_og is not None:
                for m in ['paw_l', 'paw_r', 'nose_tip']:
                    if view == 'left':
                        im = axes[i].scatter(
                            points_2d_og[view][m][n, 0], points_2d_og[view][m][n, 1],
                            s=40, c=color_orig)
                    elif view == 'right':
                        im = axes[i].scatter(
                            points_2d_og[view][m][n, 0], points_2d_og[view][m][n, 1],
                            s=40, c=color_orig)
                    ims_curr.append(im)

            if i == 1 and points_2d_reproj is not None:
                im = axes[i].scatter(
                    0.05 * img_width, 0.95 * img_height, s=40, c=color_reproj)
                ims_curr.append(im)
            if i == 1 and points_2d_og is not None:
                im = axes[i].scatter(
                    0.05 * img_width, 0.85 * img_height, s=40, c=color_orig)
                ims_curr.append(im)

        ims.append(ims_curr)

    print('creating animation...', end='')
    ani = animation.ArtistAnimation(fig, ims, blit=True, repeat=False)
    print('done')
    print('saving video to %s...' % filename, end='')
    writer = FFMpegWriter(fps=framerate, bitrate=-1)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    ani.save(filename, writer=writer)
    print('done')
