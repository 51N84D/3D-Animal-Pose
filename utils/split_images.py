#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 12:33:08 2020

@author: danbiderman
from terminal, run 
python3 split_images.py --dataset_path="../../Video_Datasets/Sawtell-data"

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
parser.add_argument("--dataset_path", default=None, help="str with folder path")
args, _ = parser.parse_known_args()


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def split_images(
    raw_images, height_lims, width_lims, is_path=False, write_path=None,
):
    assert write_path is not None
    write_path = Path(write_path).resolve()
    write_path.mkdir(exist_ok=True, parents=True)

    views = len(width_lims)
    images_split = []
    path_images = []

    for v in range(views):
        view_dir = write_path / f"view_{v}"
        view_dir.mkdir(exist_ok=True, parents=True)

        images_split.append([])
        path_images.append([])

        for idx, i in tqdm(enumerate(raw_images)):
            if is_path:
                if "png" not in i or "jpg" not in i:
                    continue
                img = cv2.imread(os.path.join(raw_img_path, i), 0)
            else:
                img = i
            view_img = img[
                height_lims[v][0] : height_lims[v][1],
                width_lims[v][0] : width_lims[v][1],
            ]
            cv2.imwrite(str(view_dir / f"{idx}.png"), view_img)
            path_images[v].append(str(view_dir / f"{idx}.png"))
            images_split[v].append(view_img)

    return images_split, path_images


if __name__ == "__main__":
    dataset_path = Path(args.dataset_path).resolve()
    raw_img_path = os.path.join(dataset_path, "frames")  # was previously 'raw_images'

    config = commentjson.load(
        open(os.path.join(dataset_path, "image_settings.json"), "r")
    )

    height_lims = config["height_lims"]
    width_lims = config["width_lims"]
    views = len(width_lims)

    images = os.listdir(raw_img_path)
    if ".DS_Store" in images:
        images.remove(".DS_Store")
    images = sorted_nicely(images)
    split_images(images, height_lims, width_lims, False, write_images=True)
