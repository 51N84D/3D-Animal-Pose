#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 12:33:08 2020

@author: danbiderman
from terminal, run 
python3 preprocess_sub_images.py --dataset_path="../../Video_Datasets/Sawtell-data"

assuming that a path to a folder is given. 
and that within that folder, we have a config json file and a raw_images folder

"""

from pathlib import Path
import os
import cv2
import re
from tqdm import tqdm
import argparse
import json
import commentjson

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", default=None, help='str with folder path')
args, _ = parser.parse_known_args()


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


if __name__ == "__main__":
    dataset_path = Path(args.dataset_path).resolve()
    raw_img_path = os.path.join(dataset_path, 'raw_images')
      
    config = commentjson.load(open(os.path.join(dataset_path, 'image_settings.json'),
                                   'r'))

     
    height_lims = config["height_lims"]
    width_lims = config["width_lims"]
    views = len(width_lims)
    
    images = os.listdir(raw_img_path)
    if ".DS_Store" in images:
        images.remove(".DS_Store")
    images = sorted_nicely(images)
    
    for v in range(views):
        view_path = os.path.join(dataset_path, 'view_%i' %v)
    
        if not os.path.isdir(view_path):
            os.mkdir(view_path)
        
        for i in tqdm(images):
            if "png" not in i:
                continue
            img = cv2.imread(os.path.join(raw_img_path,i), 0)
            view_img = img[height_lims[v][0]:height_lims[v][1], 
                           width_lims[v][0]:width_lims[v][1]]
            cv2.imwrite(os.path.join(view_path, i), view_img)