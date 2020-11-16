# Combine images
from pathlib import Path
from PIL import Image
import argparse
import os
import re
from tqdm import tqdm
from utils_IO import write_video


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    if ".DS_Store" in l:
        l.remove(".DS_Store")
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def get_args():
    parser = argparse.ArgumentParser(description="causal-gen")

    # dataset
    parser.add_argument("--frame_paths", type=str, nargs="+", required=True)
    parser.add_argument("--ind_start", type=int, default=0)
    parser.add_argument("--ind_end", type=int, required=False)
    parser.add_argument("--fps", type=int, default=5)

    return parser.parse_args()


def combine_frames(
    frame_paths,
    combined_dir="./combined",
    ind_start=0,
    ind_end=None,
    video_dir=None,
    fps=5,
):
    # Create dir to save combined images
    combined_dir = Path(combined_dir)
    combined_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(frame_paths) - 1):
        assert len(sorted_nicely(os.listdir(frame_paths[i]))) == len(
            sorted_nicely(os.listdir(frame_paths[i + 1]))
        )

    if ind_end is None:
        ind_end = len(sorted_nicely(os.listdir(frame_paths[0])))

    # Create frames
    nested_frames = []
    for frame_path in frame_paths:
        nested_frames.append(
            [Path(frame_path) / i for i in sorted_nicely(os.listdir(frame_path))]
        )

    for i in tqdm(range(ind_start, ind_end)):
        total_width = 0
        total_height = 0
        images = []
        for frames in nested_frames:
            img = Image.open(frames[i])
            images.append(img)
            width, height = img.size
            if height > total_height:
                total_height = height
            total_width += width

        new_im = Image.new("RGB", (total_width, total_height))
        curr_width = 0
        for img in images:
            new_im.paste(img, (curr_width, 0))
            curr_width, height = img.size

        new_im.save(str(combined_dir / f"combined_{i}.png"))

    if video_dir is not None:
        write_video(image_dir=combined_dir, out_file="./combined.mp4", fps=fps)


if __name__ == "__main__":
    args = get_args()
    assert args.frame_paths is not None
    combine_frames(
        args.frame_paths,
        video_dir="./",
        ind_start=args.ind_start,
        ind_end=args.ind_end,
        fps=args.fps,
    )
