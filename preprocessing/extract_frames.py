import cv2
from pathlib import Path


video_path = Path(
    "/Users/Sunsmeister/Desktop/Research/Brain/MultiView/3D-Animal-Pose/data/Sawtell-data/fish_tracking/videoEOD_cropped000.avi"
)
frame_dir = video_path.parent / Path('frames')
frame_dir.mkdir(exist_ok=True, parents=True)
vidcap = cv2.VideoCapture(str(video_path))

success, image = vidcap.read()
count = 0
while success:
    success, image = vidcap.read()
    if success:
        cv2.imwrite(str(frame_dir / f'{count}.jpg'), image) 
    count += 1
