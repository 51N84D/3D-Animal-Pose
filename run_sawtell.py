from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.resolve()))
import os
from reconstruct_points import reconstruct_points
from preprocessing.process_config import read_yaml
from addict import Dict
from ruamel.yaml import YAML
from shutil import copyfile
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="3d reconstruction")
    parser.add_argument(
        "--root_dir", type=str, default="/Volumes/sawtell-locker/C1/free/vids/"
    )
    parser.add_argument("--config", type=str, default="./configs/sawtell.yaml")
    return parser.parse_args()


if __name__ == "__main__":

    yaml = YAML()
    args = get_args()

    root_dir = Path(args.root_dir)
    config_path = Path(args.config).resolve()
    print(f"Using config {config_path}")
    output_dir = Path("./sawtell_reconstruction")  # /20201102_Joao
    video_sessions = os.listdir(root_dir)

    with open(config_path, "r") as f:
        config = yaml.load(f)

    new_config_path = Path("./configs/sawtell_iterating.yaml").resolve()
    copyfile(config_path, new_config_path)

    """
    sessions_to_keep = [
        "20201110_Joao",
        "20201120_Joao",
        "20201218_Neil",
        "20201224_Neil",
        "20201114_Greg",
        "20201120_Greg",
    ]
    """
    sessions_to_keep = []

    for session in video_sessions:
        if len(sessions_to_keep) > 0:
            keep_session = False
            for session_to_keep in sessions_to_keep:
                if session_to_keep in session:
                    keep_session = True

            if not keep_session:
                continue

        print(f"Reconstructing {session}")
        # dlc_file = /Volumes/sawtell-locker/C1/free/vids/20201102_Joao/concatenated_tracking.csv
        session_path = Path(root_dir / session)
        if "concatenated_tracking.csv" not in os.listdir(session_path):
            continue

        config["path_to_csv"] = [str(root_dir / session / "concatenated_tracking.csv")]
        config["path_to_videos"] = [str(root_dir / session / "concatenated.avi")]

        with open(new_config_path, "w") as output_stream:
            yaml.dump(config, output_stream)

        try:
            reconstruct_points(
                config=str(new_config_path),
                output_dir=str(output_dir / session),
                num_ba_frames=10000,
                num_triang_frames=1000,
                downsample=1,
                csv_type="sawtell",
            )
        except:
            print(f"Could not perform 3D reconstruction for session {session}")
            continue