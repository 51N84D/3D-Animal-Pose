from pathlib import Path
import numpy as np
import os

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
    return XYs, masks


points_path = Path('/Volumes/paninski-locker/data/ibl/raw_data/cortexlab/Subjects/KS023/2019-12-10/001/alf')

left_XYs, masks = get_markers(str(points_path), 'left')
right_XYs, masks = get_markers(str(points_path), 'right')

print('left keys: ', left_XYs.keys())
print('right keys: ', right_XYs.keys())
bodyparts = left_XYs.keys()

multiview_arr = [[], []]
for bp in bodyparts:
    print('left_XYs: ', left_XYs[bp].shape)
    print('right_XYs: ', right_XYs[bp].shape)

    multiview_arr[0].append(left_XYs[bp][:1000, :])
    multiview_arr[1].append(right_XYs[bp][:1000, :])

multiview_arr = np.asarray(multiview_arr)
multiview_arr = multiview_arr.transpose((0, 2, 1, 3))
np.save('./ibl_long.npy', multiview_arr)