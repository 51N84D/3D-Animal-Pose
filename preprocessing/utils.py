#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 12:30:42 2020

@author: danbiderman
"""
from PIL import Image
def get_image_dims(path_images):
    """loops over cameras, reads the first image, and saves dims
    num_cameras: integer
    path_images: list of lists [camera][image]"""
    img_heights = []
    img_widths = []
    num_cameras = len(path_images)
    for i in range(num_cameras):
        img = Image.open(path_images[i][0]) # read only the first image
        width, height = img.size 
        img_heights.append(height)
        img_widths.append(width)
    return img_heights, img_widths


def get_pixel_focal_lengths(focal_length_mm, 
                            sensor_size_mm, 
                            img_heights, 
                            img_widths):
    """
    computes focal length in pixels from focal length (mm) and sensor size (mm),
    for each camera.

    Args:
        focal_length_mm: array/list of length num_cameras
        sensor_size_mm: array/list of length num_cameras
        img_heights: array/list of length num_cameras
        img_widths: array/list of length num_cameras
    Returns:
        focal_lengths: array/list of length num_cameras, 
            focal length in pix per cam"""
    
    assert(len(focal_length_mm)==len(sensor_size_mm))
    # Get focal lengths
    focal_lengths = []
    num_cameras = len(focal_length_mm)
    for i in range(num_cameras):
        focal_length = (
            focal_length_mm[i] * ((img_heights[i] ** 2 + img_widths[i] ** 2) ** (1 / 2))
        ) / sensor_size_mm[i]
        focal_lengths.append(focal_length)
    return focal_lengths