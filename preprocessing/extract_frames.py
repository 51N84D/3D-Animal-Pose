### script that extracts frames from a video, written by Sun and edited by Dan.
'''To run in terminal, when in 3D-Animal-Pose main folder, run
3D-Animal-Pose danbiderman$ python3 preprocessing/extract_frames.py --video_path='/Volumes/sawtell-locker/C1/free/vids/20201102_Joao/concatenated.avi' --frames_path='../Video_Datasets/Sawtell-data/20201102_Joao/frames' --max_frames=25000 --downsample=5'''


import cv2
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", default=None, help='str with video path')
parser.add_argument("--frames_path", default=None, help='str with path for saving frames')
parser.add_argument("--max_frames", default=100, type = int, help='int with max frames starting from zero')
parser.add_argument("--downsample", default=1, type = int, help='int with num_frames to downsample')
args, _ = parser.parse_known_args()


video_path = Path(args.video_path)
frame_dir = Path(args.frames_path)

print('Starting to save the frames of %s in the folder %s' % (str(video_path), str(frame_dir)))
frame_dir.mkdir(exist_ok=True, parents=True)
vidcap = cv2.VideoCapture(str(video_path))

downsampling = args.downsample
global_count = 0
local_count = 0
success, image = vidcap.read()
cv2.imwrite(str(frame_dir / f'{local_count}.png'), image) # save first frame
global_count += 1
local_count += 1

while success:
    success, image = vidcap.read()
    if success and global_count % downsampling == 0:
        cv2.imwrite(str(frame_dir / f'{local_count}.png'), image)
        local_count += 1
    global_count += 1
    if global_count == args.max_frames:
        break
