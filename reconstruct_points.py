from pathlib import Path
import argparse
from preprocessing.process_config import read_yaml
from utils.utils_BA import clean_nans
import numpy as np
from utils.points_to_dataframe import points3d_arr_to_df


def get_args():
    parser = argparse.ArgumentParser(description="3d reconstruction")

    # dataset
    parser.add_argument("--config", type=str, default="./configs/sawtell.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    experiment_data = read_yaml(args.config)
    cam_group = experiment_data["cam_group"]
    num_frames = experiment_data["num_frames"]
    num_bodyparts = experiment_data["num_bodyparts"]
    config = experiment_data["config"]

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
    if len(points_3d.shape) < 3:
        points_3d = points_3d.reshape(num_frames, num_bodyparts, 3)
    # Save points
    points_dir = Path('./points').resolve()
    points_dir.mkdir(parents=True, exist_ok=True)
    
    df = points3d_arr_to_df(points_3d, config.bp_names)
    df.to_csv(points_dir / f'{Path(args.config).stem}_3d.csv')
    print("Finished saving points.")
