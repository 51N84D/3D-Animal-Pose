from __future__ import print_function
import os
import numpy as np
from pathlib import Path, PurePath
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.resolve()))
import csv
import argparse
from utils.utils_IO import sorted_alphanumeric
from PIL import Image
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Sawtell preprocessing")
    parser.add_argument(
        "--dataset", type=str, default="./data/Sawtell-data/tank_dataset_11.h5"
    )
    parser.add_argument(
        "--image_settings", type=str, default="./data/Sawtell-data/image_settings.json"
    )
    return parser.parse_args()


def get_data():
    import h5py

    args = get_args()
    filename = Path(args.dataset)
    img_settings_file = Path(args.image_settings)

    f = h5py.File(filename, "r")

    print("-------------DATASET INFO--------------")
    print("keys: ", f.keys())
    print("annotated: ", f["annotated"])
    print("annotations: ", f["annotations"])
    print("images: ", f["images"])
    print("skeleton: ", f["skeleton"])
    print("skeleton_names: ", f["skeleton_names"])
    print("frame_number: ", f["frame_number"])
    print("video_name: ", f["video_name"])
    print("----------------------------------------")

    # for i, fn in enumerate(f['frame_number']):
    #    print('video: ', f['video_name'][i])
    #    print('frame: ', fn)

    # Read frame limits:
    import commentjson

    img_settings = commentjson.load(open(str(img_settings_file), "r"))
    num_cameras = len(img_settings["height_lims"])

    # Get points from hd5
    pts_array = np.asarray(f["annotations"])

    # Get number of bodyparts
    num_analyzed_body_parts = int(pts_array.shape[1] / num_cameras)

    # Split points into separate views
    multiview_idx_to_name = {}
    multiview_name_to_idx = {}

    # NOTE: the order should match the order in `image_settings.json`
    view_names = ["top", "main", "right"]
    assert len(view_names) == num_cameras

    # NOTE: Empty list keeps all bodyparts
    # bp_to_keep = ["head", "mid", "pectoral"]  # ["head", "chin"]
    # bp_to_keep = ["chin"]
    bp_to_keep = []

    for view_name in view_names:
        multiview_name_to_idx[view_name] = []

    new_skeleton_names = []
    for idx, name in enumerate(f["skeleton_names"]):
        if len(bp_to_keep) > 0:
            skip_bp = True
            for bp in bp_to_keep:
                if bp == name.decode("UTF-8").split("_")[0]:
                    skip_bp = False
            if skip_bp:
                continue

        new_skeleton_names.append(name)
        for view_name in view_names:
            if view_name in name.decode("UTF-8").split("_")[-1]:
                multiview_idx_to_name[idx] = view_name
                multiview_name_to_idx[view_name].append(idx)

    if len(bp_to_keep) > 0:
        num_analyzed_body_parts = int(len(new_skeleton_names) / num_cameras)

    # (num views, num frames, num points per frame, 2)
    multiview_pts_2d = np.empty(
        shape=(num_cameras, pts_array.shape[0], num_analyzed_body_parts, 2)
    )
    for i, view_name in enumerate(view_names):
        # Select rows from indices
        view_indices = multiview_name_to_idx[view_name]
        view_points = pts_array[:, view_indices, :]
        view_points[:, :, 0] -= img_settings["width_lims"][i][0]
        view_points[:, :, 1] -= img_settings["height_lims"][i][0]
        multiview_pts_2d[i, :, :, :] = view_points

    # Now, convert to (num views, num points * num frames, 2)
    multiview_pts_2d = np.reshape(
        multiview_pts_2d,
        (
            multiview_pts_2d.shape[0],
            multiview_pts_2d.shape[1] * multiview_pts_2d.shape[2],
            -1,
        ),
    )

    assert multiview_pts_2d.shape[-1] == 2

    bp_row = ["bodyparts"]
    coord_row = ["coords"]
    bodyparts = []
    for name in list(f["skeleton_names"][:]):
        bp_row.append(name.decode("UTF-8"))
        bp_row.append(name.decode("UTF-8"))
        bodyparts.append(name.decode("UTF-8"))
        coord_row.append("x")
        coord_row.append("y")

    scorer = "sun"
    # Can try construction each row separately
    scorer_row = ["scorer"] + [scorer] * (len(bp_row) - 1)
    
    with open('./bodyparts.txt', 'w') as txt_f:
        for bp in bodyparts:
            txt_f.write("%s\n" % bp)
    # xy_coords = list(zip(pts_array[0, :, 0], pts_array[0, :, 1]))
    # xy_coords_flat = [item for sublist in xy_coords for item in sublist]

    # Get max frame:
    max_frame = 0
    for fn in f["frame_number"]:
        if fn > max_frame:
            max_frame = fn

    with tqdm(total=len(f['images'])) as pbar:
        for i, img in enumerate(f["images"]):
            xy_coords = list(zip
            (pts_array[i, :, 0], pts_array[i, :, 1]))
            xy_coords_flat = [item for sublist in xy_coords for item in sublist]
            full_vid_path = PurePath(
                (f["video_name"][i].decode("UTF-8").split(":")[-1]).replace("\\", "/")
            )
            dir_name = full_vid_path.parent.name
            if dir_name == '':
                pbar.update(1)
                continue
            label_dir = Path("./labeled-data") / dir_name
            label_dir.mkdir(parents=True, exist_ok=True)
            frame_name = f"{f['frame_number'][i]:0{len(str(max_frame))}d}"

            label_path = label_dir / ('img' + frame_name + ".png")
            # Write image to label path:
            pil_image = Image.fromarray(np.squeeze(img))
            pil_image.save(label_path)
            if os.path.isfile(label_dir / f"CollectedData_{scorer}.csv"):
                with open(label_dir / f"CollectedData_{scorer}.csv", "a") as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([label_path] + xy_coords_flat)
            else:
                with open(label_dir / f"CollectedData_{scorer}.csv", "w") as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(scorer_row)
                    csvwriter.writerow(bp_row)
                    csvwriter.writerow(coord_row)
                    csvwriter.writerow([label_path] + xy_coords_flat)
            pbar.update(1)


if __name__ == "__main__":
    get_data()