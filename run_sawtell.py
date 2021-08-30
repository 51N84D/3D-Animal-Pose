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
from datetime import datetime


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
    # output_dir = Path("./sawtell_reconstruction")  # /20201102_Joao
    output_dir = Path("d:/github/DN_local_repo/sawtell_reconstruction")
    video_sessions = sorted(os.listdir(root_dir))

    with open(config_path, "r") as f:
        config = yaml.load(f)

    new_config_path = Path("./configs/sawtell_iterating.yaml").resolve()
    copyfile(config_path, new_config_path)

    sessions_to_keep_dict = {
        "Fred": (datetime(2020, 9, 7), datetime(2020, 10, 1)),
        "Greg": (datetime(2020, 11, 4), datetime(2020, 11, 22)),
        # "Greg": (datetime(2020, 11, 10), datetime(2020, 11, 10)),
        "Igor": (datetime(2020, 11, 4), datetime(2020, 11, 22)),
        "Joao": (datetime(2020, 11, 4), datetime(2020, 11, 22)),
        "Kyle": (datetime(2020, 11, 21), datetime(2020, 12, 11)),
        "Lazy": (datetime(2020, 11, 21), datetime(2020, 12, 11)),
        "Mark": (datetime(2020, 12, 1), datetime(2020, 12, 18)),
        "Neil": (datetime(2020, 12, 12), datetime(2020, 12, 30)),
        "Omar": (datetime(2020, 12, 12), datetime(2020, 12, 30)),
        "Raul": (datetime(2021, 1, 7), datetime(2021, 1, 24)),
        "Sean": (datetime(2021, 1, 7), datetime(2021, 1, 24)),
    }
    # 20201110_Greg

    for session in video_sessions:
        keep_session = False

        if "_" in session:
            fish_name = session.split("_")[-1]
            session_datestring = session.split("_")[0]
            if fish_name in sessions_to_keep_dict:
                # Process date into datetime
                sess_year = int(session_datestring[:4])
                sess_month = int(session_datestring[4:6])
                sess_day = int(session_datestring[6:])
                session_datetime = datetime(sess_year, sess_month, sess_day)
                if (
                    session_datetime >= sessions_to_keep_dict[fish_name][0]
                    and session_datetime <= sessions_to_keep_dict[fish_name][1]
                ):
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

        # try:
        reconstruct_points(
            config=str(new_config_path),
            output_dir=str(output_dir / session),
            num_ba_frames=10000,
            num_triang_frames=1000,
            downsample=1,
            csv_type="sawtell",
            nrows=1000,
        )

        """
        except:
            print(f"Could not perform 3D reconstruction for session {session}")
            continue
        """
