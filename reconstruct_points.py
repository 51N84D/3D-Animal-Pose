from pathlib import Path
import argparse
from preprocessing.process_config import read_yaml
from utils.utils_BA import clean_nans
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="3d reconstruction")

    # dataset
    parser.add_argument("--config", type=str, default="./configs/sawtell.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    experiment_data = read_yaml(args.config)
    cam_group = experiment_data["cam_group"]
    # Bundle adjust points
    pts_2d_joints = experiment_data["points_2d_joints"]
    pts_2d = pts_2d_joints.reshape(
        pts_2d_joints.shape[0],
        pts_2d_joints.shape[1] * pts_2d_joints.shape[2],
        pts_2d_joints.shape[3],
    )
    pts_2d_filtered, clean_point_indices = clean_nans(pts_2d)
    res, points_3d_init = cam_group.bundle_adjust(pts_2d_filtered)

    # Triangulate points
    points_3d = cam_group.triangulate_progressive(pts_2d_joints)
    
    # Save points
    points_dir = Path('./points').resolve()
    points_dir.mkdir(parents=True, exist_ok=True)
    np.save(points_dir / f'{Path(args.config).stem}_3d.npy', points_3d)
    print("Finished saving points.")
