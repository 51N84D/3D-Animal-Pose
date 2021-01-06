import cv2
from pathlib import Path


# video_path = Path(
#     "/Users/Sunsmeister/Desktop/Research/Brain/MultiView/3D-Animal-Pose/data/Sawtell-data/fish_tracking/videoEOD_cropped000.avi"
# )

video_path = Path('/Volumes/paninski-locker/data/ibl/raw_data/cortexlab/Subjects/KS023/2019-12-10/001/raw_video_data/_iblrig_leftCamera.raw.mp4')
# frame_dir = video_path.parent / Path('frames')
frame_dir = Path('./left_frames')
frame_dir.mkdir(exist_ok=True, parents=True)
vidcap = cv2.VideoCapture(str(video_path))

downsampling = 1
global_count = 0
local_count = 0
success, image = vidcap.read()
cv2.imwrite(str(frame_dir / f'{local_count}.png'), image)
global_count += 1
local_count += 1

while success:
    success, image = vidcap.read()
    if success and global_count % downsampling == 0:
        cv2.imwrite(str(frame_dir / f'{local_count}.png'), image)
        local_count += 1
    global_count += 1
    if global_count == 100:
        break
