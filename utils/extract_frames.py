import cv2
import numpy as np


def extract_frames(indices, video_path, exact=True):
    cap = cv2.VideoCapture(video_path)
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frames = []
    if exact:
        indices_dict = dict((idx, 0) for idx in indices)
        curr_idx = 0
        while len(indices_dict.keys()) > 0 and curr_idx < totalFrames:
            ret, frame = cap.read()
            if curr_idx in indices_dict:
                frames.append(frame)
                del indices_dict[curr_idx]
            curr_idx += 1

    else:
        for frame_idx in indices:
            # get total number of frames
            # check for valid frame number
            if frame_idx >= 0 & frame_idx <= totalFrames:
                # set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                frames.append(frame)
    return frames


"""
frame_indices = np.arange(20000)
print("frame_indices: ", frame_indices)
video_path = (
    "/Volumes/sawtell-locker/C1/free/vids/20201102_Joao/concatenated_tracking.avi"
)

frames = extract_frames(frame_indices, video_path)

frames = np.asarray(frames)
print(frames.shape)
np.save("sawtell_frames.npy", frames)
"""

