#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:00:04 2021

@author: danbiderman
"""
import numpy as np

def k_largest_inds(vec, k):
    '''the indices of the k largest values.
    np.argsort() sorts from low to high, therefore we negate it'''
    assert k < len(vec)
    return (-vec).argsort()[:k]

def slice_high_confidence(pts, confs, k):
    assert pts.shape[:-1] == confs.shape 
    mean_confs = confs.mean(axis=(0, 2))
    top_conf_frames = pts[:, 
                        k_largest_inds(mean_confs, k), :, :]
    assert top_conf_frames.shape == (pts.shape[0], k, pts.shape[2], pts.shape[3])
    return top_conf_frames