from __future__ import print_function
import os
import numpy as np
import re
from ibl_utils import get_markers
import cv2
from pathlib import Path
from copy import deepcopy
import argparse
from tqdm import tqdm


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def get_args():
    parser = argparse.ArgumentParser(description="ibl preprocessing")

    # dataset
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        default="/Volumes/paninski-locker/data/ibl/raw_data/",
    )
    parser.add_argument(
        "--save_points_path",
        type=str,
        default="/Users/Sunsmeister/Desktop/Research/Brain/MultiView/3D-Animal-Pose/data/ibl.npy",
    )
    parser.add_argument(
        "--save_lh_path",
        type=str,
        default="/Users/Sunsmeister/Desktop/Research/Brain/MultiView/3D-Animal-Pose/data/ibl_lh.npy",
    )
    parser.add_argument(
        "--save_frame_path",
        type=str,
        default="/Users/Sunsmeister/Desktop/Research/Brain/MultiView/3D-Animal-Pose/data/IBL_full/images/",
    )
    parser.add_argument("--save_lh", action="store_true", help="save likelihoods")
    parser.add_argument("--write_frames", action="store_true", help="write frames")
    parser.add_argument(
        "--occlusions",
        action="store_true",
        help="filter points to find occlusions",
    )
    parser.add_argument(
        "--likelihood_thresh",
        default=0.9,
        type=float,
        help="threshold for filtering points",
    )
    parser.add_argument("--start_frame", default=0, type=int, help="frame to start at")
    parser.add_argument(
        "--num_frames", default=1000, type=int, help="number of frames to consider"
    )
    parser.add_argument(
        "--padding",
        default=5,
        type=int,
        help="padding for before and after occluded frames",
    )

    return parser.parse_args()


def select_occluded_frames(likelihoods, threshold=0.9, padding=5):
    lh_low = likelihoods < threshold
    lh_low = np.any(lh_low[:, :, 1:], axis=-1)
    lh_low = np.any(lh_low, axis=0)

    occluded_indices = np.where(lh_low)[0]
    padded_indices = []
    occluded_dict = {}
    # Now fill using padding
    for i in tqdm(range(likelihoods.shape[1])):
        for j in range(padding):
            if (
                i + j in occluded_indices or i - j in occluded_indices
            ) and i not in occluded_dict:
                padded_indices.append(i)
                occluded_dict[i] = 0

    return padded_indices, occluded_dict


def get_data():
    args = get_args()
    raw_data_dir = Path(args.raw_data_dir)
    save_frame_path = Path(args.save_frame_path)
    save_points_path = Path(args.save_points_path)
    save_lh_path = Path(args.save_lh_path)
    num_frames = args.num_frames
    start_frame = args.start_frame

    likelihood_thresh = args.likelihood_thresh

    save_frame_path.mkdir(exist_ok=True, parents=True)

    camera = "front"
    # 'right': 150 Hz (640 x 512)
    # 'left': 60 Hz (1280 x 1024)
    # 'body': 30 Hz (640 x 512)

    lab = "cortexlab"
    animal = "KS023"
    date = "2019-12-10"
    number = "001"

    # raw data from flatiron
    session_path = raw_data_dir / lab / "Subjects" / animal / date / number
    alf_path = session_path / "alf"
    video_path = session_path / "raw_video_data"

    if camera == "all":
        views = ["right", "left", "body"]
    elif camera == "front":
        views = ["right", "left"]
    else:
        views = [camera]

    mp4_files = {view: video_path / f"_iblrig_{view}Camera.raw.mp4" for view in views}

    timestamps = {}
    for view in views:
        timestamps[view] = np.load(alf_path / f"_ibl_{view}Camera.times.npy")
        print(
            "average %s camera framerate: %f Hz"
            % (view, 1.0 / np.mean(np.diff(timestamps[view])))
        )

    markers = {}
    masks = {}
    likelihoods = {}
    print("Getting markers...")
    for view in views:
        markers[view], masks[view], likelihoods[view] = get_markers(
            alf_path, view, likelihood_thresh=likelihood_thresh
        )
        # chop off timestamps that don't have markers
        timestamps[view] = timestamps[view][: markers[view]["nose_tip"].shape[0]]
    # get closest stamps on right cam (150 Hz) for each stamp of left (60 Hz)
    from scipy.interpolate import interp1d

    interpolater = interp1d(
        timestamps["right"],
        np.arange(len(timestamps["right"])),
        kind="nearest",
        fill_value="extrapolate",
    )
    idx_aligned = np.round(interpolater(timestamps["left"])).astype(np.int)

    # ------------------------------------------------------------
    # Read and write numpy array
    print("Making array...")
    multiview_arr = []
    likelihoods_arr = []
    for view in views:
        multiview_arr.append([])
        likelihoods_arr.append([])

    bodyparts = markers[views[0]].keys()
    bp_to_keep = ["nose_tip", "paw_l", "paw_r"]

    # Swap left and right paw in one view for consistency
    paw_r = deepcopy(markers["right"]["paw_r"])
    markers["right"]["paw_r"] = markers["right"]["paw_l"]
    markers["right"]["paw_l"] = paw_r

    paw_r_lh = deepcopy(likelihoods["right"]["paw_r"])
    likelihoods["right"]["paw_r"] = likelihoods["right"]["paw_l"]
    likelihoods["right"]["paw_l"] = paw_r_lh

    for bp in bodyparts:
        if bp not in bp_to_keep:
            continue
        for i, view in enumerate(views):
            if view == "left":
                multiview_arr[i].append(
                    markers[view][bp][start_frame : start_frame + num_frames, :]
                )
                likelihoods_arr[i].append(
                    likelihoods[view][bp][start_frame : start_frame + num_frames]
                )

            elif view == "right":
                multiview_arr[i].append(
                    markers[view][bp][idx_aligned][
                        start_frame : start_frame + num_frames, :
                    ]
                )
                likelihoods_arr[i].append(
                    likelihoods[view][bp][idx_aligned][
                        start_frame : start_frame + num_frames
                    ]
                )
    multiview_arr = np.array(multiview_arr)
    multiview_arr = multiview_arr.transpose((0, 2, 1, 3))

    likelihoods_arr = np.array(likelihoods_arr)
    likelihoods_arr = likelihoods_arr.transpose((0, 2, 1))
    # ------------------------------------------------------------

    # Select occluded frames
    if args.occlusions:
        occluded_indices, keep_dict = select_occluded_frames(
            likelihoods_arr, threshold=likelihood_thresh, padding=args.padding
        )
    else:
        occluded_indices = np.arange(multiview_arr.shape[1])
        keep_dict = dict((e, i) for (i, e) in enumerate(occluded_indices))

    multiview_arr = multiview_arr[:, occluded_indices, :, :]
    likelihoods_arr = likelihoods_arr[:, occluded_indices, :]

    print("multiview_arr: ", multiview_arr.shape)
    print(len(occluded_indices))
    print(len(keep_dict.keys()))
    # Read and write video frames
    if args.write_frames:
        print("Writing frames...")
        # ------------------------------------------------------------
        for view in views:
            frame_path = save_frame_path / view
            frame_path.mkdir(exist_ok=True, parents=True)

            # Extract frames at specific timesteps
            vidcap = cv2.VideoCapture(str(mp4_files[view]))
            count = 0
            local_count = 0
            tracker_count = 0

            font = cv2.FONT_HERSHEY_SIMPLEX
            while True:
                success, image = vidcap.read()
                if success:
                    height, width = image.shape[:2]

                    # Handle right view
                    if view == "right":
                        cv2.putText(
                            image,
                            str(timestamps[view][count]),
                            (width - 300, height - 100),
                            font,
                            1,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                        if count in idx_aligned:
                            if tracker_count in keep_dict:
                                cv2.imwrite(
                                    str(frame_path / f"{local_count}.png"), image
                                )
                                local_count += 1
                            tracker_count += 1

                    # Handle left view
                    else:
                        cv2.putText(
                            image,
                            str(timestamps[view][count]),
                            (width - 400, height - 100),
                            font,
                            2,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                        if count in keep_dict:
                            cv2.imwrite(str(frame_path / f"{local_count}.png"), image)
                            local_count += 1

                        tracker_count += 1

                    count += 1

                if tracker_count >= num_frames or not success:
                    break
        # ------------------------------------------------------------

    np.save(save_points_path, multiview_arr)

    if args.save_lh:
        np.save(save_lh_path, likelihoods_arr)


if __name__ == "__main__":
    get_data()