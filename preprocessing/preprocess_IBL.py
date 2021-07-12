from __future__ import print_function
import os
import numpy as np
import re
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import pandas as pd
import cv2
from copy import deepcopy
import pdb
from scipy.interpolate import interp1d

sys.path.append(str(Path(__file__).resolve().parent.parent.resolve()))

"""
Usage: 

"""


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
        default="/Volumes/paninski-locker/data/ibl/dlc-networks/paw2-mic-2021-02-21/videos",
    )
    parser.add_argument(
        "--timestamps",
        type=str,
        default="/Volumes/paninski-locker/data/ibl/dlc-networks/paw2-mic-2021-02-21/timestamps",
    )

    parser.add_argument(
        "--likelihood_thresh",
        default=0.9,
        type=float,
        help="threshold for filtering points",
    )
    parser.add_argument(
        "--write_dir",
        type=str,
        default="./ibl_videos_aligned",
    )
    parser.add_argument(
        "--save_frames",
        action="store_true",
    )

    parser.add_argument("--select_subset", type=int, help="Number of frames to select")

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


def get_data(
    raw_data_dir,
    times_path,
    write_dir,
    flip_right=True,
    select_subset=None,
    save_frames=False,
):
    """Return pandas df and"""

    raw_data_dir = Path(raw_data_dir).resolve()
    times_path = Path(times_path)
    write_dir = Path(write_dir)
    write_dir.mkdir(exist_ok=True, parents=True)

    session_eids = [
        # "7a887357-850a-4378-bd2a-b5bc8bdd3aac",
        "6c6983ef-7383-4989-9183-32b1a300d17a",
        # "88d24c31-52e4-49cc-9f32-6adbeb9eba87",
        # "81a78eac-9d36-4f90-a73a-7eb3ad7f770b",
        # "cde63527-7f5a-4cc3-8ac2-215d82e7da26",
    ]

    if save_frames:
        frames_dir = write_dir / "frames"
        frames_dir.mkdir(exist_ok=True, parents=True)

    for session_eid in session_eids:
        if save_frames:
            session_frames = frames_dir / session_eid
            session_frames.mkdir(exist_ok=True, parents=True)
            left_session_frames = session_frames / "left"
            left_session_frames.mkdir(exist_ok=True, parents=True)
            right_session_frames = session_frames / "right"
            right_session_frames.mkdir(exist_ok=True, parents=True)

        # Exctract eid of interest
        all_videos = os.listdir(raw_data_dir)
        for vid in all_videos:
            if vid.startswith(session_eid) and vid.endswith(".csv"):
                if "left" in vid:
                    left_csv_path = raw_data_dir / vid
                elif "right" in vid and "selected" not in vid:
                    right_csv_path = raw_data_dir / vid

            if (
                vid.startswith(session_eid)
                and vid.endswith(".mp4")
                and "labeled" not in vid
            ):
                if "left" in vid:
                    left_vid_path = raw_data_dir / vid
                elif "right" in vid and "selected" not in vid:
                    right_vid_path = raw_data_dir / vid

        # Read csv into pandas
        left_points_df = pd.read_csv(left_csv_path, header=[0, 1, 2], chunksize=10000)
        left_points_df = pd.concat([i for i in tqdm(left_points_df)], ignore_index=True)
        right_points_df = pd.read_csv(right_csv_path, header=[0, 1, 2], chunksize=10000)
        right_points_df = pd.concat(
            [i for i in tqdm(right_points_df)], ignore_index=True
        )

        if select_subset is None:
            select_subset = left_points_df.shape[0]

        num_bodyparts_left = int(((left_points_df.columns.shape[0]) - 1) / 3)
        num_bodyparts_right = int(((right_points_df.columns.shape[0]) - 1) / 3)
        assert num_bodyparts_left == num_bodyparts_right
        num_bodyparts = num_bodyparts_left

        # print(left_points_df.columns.values)
        left_points_and_confs = left_points_df.values[:, 1:]
        right_points_and_confs = right_points_df.values[:, 1:]

        # Select columns with points and likelihoods seperately
        left_points_arr = np.empty((left_points_and_confs.shape[0], num_bodyparts, 2))
        left_confs = np.empty((left_points_and_confs.shape[0], num_bodyparts))
        for i in range(num_bodyparts):
            x = left_points_and_confs[:, 3 * i]
            y = left_points_and_confs[:, 3 * i + 1]
            lh = left_points_and_confs[:, 3 * i + 2]
            left_points_arr[:, i, 0] = x
            left_points_arr[:, i, 1] = y
            left_confs[:, i] = lh

        # Select columns with points and likelihoods seperately
        right_points_arr = np.empty((right_points_and_confs.shape[0], num_bodyparts, 2))
        right_confs = np.empty((right_points_and_confs.shape[0], num_bodyparts))
        for i in range(num_bodyparts):
            x = right_points_and_confs[:, 3 * i]
            y = right_points_and_confs[:, 3 * i + 1]
            lh = right_points_and_confs[:, 3 * i + 2]
            right_points_arr[:, i, 0] = x
            right_points_arr[:, i, 1] = y
            right_confs[:, i] = lh

        # Get timestamps
        timestamp_list = os.listdir(times_path)
        for time_path in timestamp_list:
            if session_eid in time_path:
                if "left" in time_path:
                    left_time_path = times_path / time_path
                    times_left = np.load(left_time_path)
                elif "right" in time_path:
                    right_time_path = times_path / time_path
                    times_right = np.load(right_time_path)

        left_points_arr = left_points_arr[:select_subset]
        times_left = times_left[: left_points_arr.shape[0]]
        times_right = times_right[: right_points_arr.shape[0]]

        interpolater = interp1d(
            times_right,
            np.arange(len(times_right)),
            kind="nearest",
            fill_value="extrapolate",
        )

        idx_aligned = np.round(interpolater(times_left[:select_subset])).astype(np.int)
        right_points_arr = right_points_arr[idx_aligned, :, :]
        right_confs = right_confs[idx_aligned, :]

        # Make new videos and plot random frames
        left_cap = cv2.VideoCapture(str(left_vid_path))
        left_fps = left_cap.get(cv2.CAP_PROP_FPS)

        right_cap = cv2.VideoCapture(str(right_vid_path))
        right_fps = right_cap.get(cv2.CAP_PROP_FPS)

        success, image = right_cap.read()
        height, width, layers = image.shape
        right_size = (width, height)

        right_out_vid = cv2.VideoWriter(
            filename=str(write_dir / (right_vid_path.stem + "_selected.mp4")),
            fourcc=cv2.VideoWriter_fourcc("m", "p", "4", "v"),
            fps=left_fps,
            frameSize=right_size,
        )
        count = 0

        while success:
            if count in idx_aligned:
                for _ in range(np.count_nonzero(idx_aligned == count)):
                    right_out_vid.write(np.flip(image, axis=1))
                    if save_frames:
                        cv2.imwrite(str(right_session_frames / f"{count}.png"), image)
            success, image = right_cap.read()
            count += 1

        success, image = left_cap.read()
        height, width, layers = image.shape
        left_size = (width, height)

        left_out_vid = cv2.VideoWriter(
            filename=str(write_dir / (left_vid_path.stem + "_selected.mp4")),
            fourcc=cv2.VideoWriter_fourcc("m", "p", "4", "v"),
            fps=left_fps,
            frameSize=left_size,
        )
        
        count = 0
        while success:
            left_out_vid.write(image)
            success, image = left_cap.read()
            if save_frames:
                cv2.imwrite(str(left_session_frames / f"{count}.png"), image)
            count += 1
            if count >= select_subset:
                break

        left_points_df = left_points_df[
            left_points_df.index.isin(np.arange(select_subset))
        ]
        # right_points_df = right_points_df[right_points_df.index.isin(idx_aligned)]
        right_points_df = right_points_df.reindex(idx_aligned)

        # Flip points
        for i in range(num_bodyparts):
            right_points_df.iloc[:, 1 + i * 3] = (
                right_size[0] - right_points_df.iloc[:, 1 + i * 3]
            )

        # Swap labels
        right_points_df_swapped = deepcopy(right_points_df)
        right_points_df_swapped.iloc[:, 1] = right_points_df.iloc[:, 4]
        right_points_df_swapped.iloc[:, 4] = right_points_df.iloc[:, 1]
        right_points_df_swapped.iloc[:, 2] = right_points_df.iloc[:, 5]
        right_points_df_swapped.iloc[:, 5] = right_points_df.iloc[:, 2]
        right_points_df_swapped.iloc[:, 3] = right_points_df.iloc[:, 6]
        right_points_df_swapped.iloc[:, 6] = right_points_df.iloc[:, 3]

        left_points_df_swapped = deepcopy(left_points_df)
        left_points_df_swapped.iloc[:, 1] = left_points_df.iloc[:, 4]
        left_points_df_swapped.iloc[:, 4] = left_points_df.iloc[:, 1]
        left_points_df_swapped.iloc[:, 2] = left_points_df.iloc[:, 5]
        left_points_df_swapped.iloc[:, 5] = left_points_df.iloc[:, 2]
        left_points_df_swapped.iloc[:, 3] = left_points_df.iloc[:, 6]
        left_points_df_swapped.iloc[:, 6] = left_points_df.iloc[:, 3]

        with open(str(write_dir / f"{left_vid_path.stem}.csv"), "w") as file:
            left_points_df_swapped.to_csv(file, index=False)

        with open(str(write_dir / f"{right_vid_path.stem}.csv"), "w") as file:
            right_points_df.to_csv(file, index=False)


if __name__ == "__main__":
    args = get_args()
    get_data(args.raw_data_dir, args.timestamps, args.write_dir, save_frames=args.save_frames)